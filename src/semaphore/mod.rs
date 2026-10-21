//! A futures-aware semaphore used to maintain the ACID compliance of a mutable data store.
//!
//! This differs from [`tokio::sync::Semaphore`] by tracking read and write reservations
//! rather than a single atomic integer. This is because of the additional logical constraints of
//! a transactional resource--
//! a write permit blocks future read permits and a read permit conflicts with past write permits.
//!
//! More information: [https://en.wikipedia.org/wiki/ACID](https://en.wikipedia.org/wiki/ACID)
//!
//! Example:
//! ```
//! use std::ops::Range;
//!
//! use collate::Collator;
//!
//! use txn_lock::semaphore::*;
//! use txn_lock::Error;
//!
//! let collator = Collator::default();
//! let semaphore = Semaphore::<u64, Collator<usize>, Range<usize>>::new(collator);
//!
//! // Multiple overlapping read permits within a transaction are fine
//! let permit0_1 = semaphore.try_read(0, 0..1).expect("read 0..1");
//! let permit0_2 = semaphore.try_read(0, 0..1).expect("read 0..1");
//! // And non-overlapping write permits are fine
//! let permit0_3 = semaphore.try_write(0, 2..3).expect("write 2..3");
//! // But an overlapping write permit in the same transaction will block
//! assert_eq!(semaphore.try_write(0, 0..2).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(permit0_1);
//! assert_eq!(semaphore.try_write(0, 0..2).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(permit0_2);
//! // Until all overlapping permits are dropped
//! let permit0_3 = semaphore.try_write(0, 0..2).expect("write 0..2");
//!
//! // Finalizing a transaction will un-block permits for later transactions
//! // It's the caller's responsibility to make sure that data can't be mutated after finalizing
//! semaphore.finalize(&0, false);
//!
//! // Now permits for later transactions are un-blocked
//! let permit1 = semaphore.try_read(1, 1..2).expect("permit");
//! // Acquiring a write permit is fine even if there's a read permit in the past
//! let permit2 = semaphore.try_write(2, 1..3).expect("permit");
//! // But it will block all permits for later transactions
//! assert_eq!(semaphore.try_read(3, 1..4).unwrap_err(), Error::WouldBlock);
//!
//! // To prevent a memory leak, finalize all transactions earlier than the given ID
//! semaphore.finalize(&2, true);
//!
//! // Now later permits are un-blocked
//! let permit3 = semaphore.try_write(3, 1..4).expect("permit");
//! // And trying to write-lock the past will result in a conflict error
//! assert_eq!(semaphore.try_write(2, 2..3).unwrap_err(), Error::Conflict);
//!
//! // It's still allowed to acquire a write permit for later transactions
//! // if the range doesn't overlap with any other reservations
//! let permit4 = semaphore.try_write(4, 4..5).expect("permit");
//! ```
//!
//! [`tokio::sync::Semaphore`]: https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use collate::{Collate, Overlap, OverlapsRange};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore as NativeSemaphore};

use super::{Error, Result};

const PERMITS: u32 = u32::MAX >> 3;

struct Reservation<R> {
    range: Arc<R>,
    write: bool,
    semaphore: Arc<NativeSemaphore>,
}

impl<R> Reservation<R> {
    fn new(range: Arc<R>, write: bool) -> Self {
        Self {
            range,
            write,
            semaphore: Arc::new(NativeSemaphore::new(PERMITS as usize)),
        }
    }
}

#[derive(Debug)]
struct Permit<R> {
    range: Arc<R>,
    owned: Option<OwnedSemaphorePermit>,
    notify: Arc<Notify>,
}

impl<R> Drop for Permit<R> {
    fn drop(&mut self) {
        // A woken task must observe the released capacity, even on another thread.
        drop(self.owned.take());
        self.notify.notify_waiters();
    }
}

/// A permit to read a specific section of a transactional resource.
#[derive(Debug)]
pub struct PermitRead<R> {
    permit: Permit<R>,
}

impl<R> Deref for PermitRead<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        &self.permit.range
    }
}

/// A permit to write to a specific section of a transactional resource.
#[derive(Debug)]
pub struct PermitWrite<R> {
    permit: Permit<R>,
}

impl<R> Deref for PermitWrite<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        &self.permit.range
    }
}

enum Attempt<I, R> {
    Acquired(Permit<R>),
    Pending(I, Arc<R>),
}

/// A semaphore maintaining transactional ordering and overlapping-range exclusion.
///
/// Acquisition scans exact reservations under one short-lived mutex; permits never
/// hold that mutex during caller work. Async retries are non-FIFO and may barge:
/// starvation freedom is not guaranteed.
pub struct Semaphore<I, C, R> {
    collator: C,
    versions: Arc<Mutex<BTreeMap<I, Vec<Reservation<R>>>>>,
    notify: Arc<Notify>,
}

impl<I, C: Clone, R> Clone for Semaphore<I, C, R> {
    fn clone(&self) -> Self {
        Self {
            collator: self.collator.clone(),
            versions: self.versions.clone(),
            notify: self.notify.clone(),
        }
    }
}

impl<I, C, R> Semaphore<I, C, R> {
    /// Construct a new transactional [`Semaphore`].
    pub fn new(collator: C) -> Self {
        Self {
            collator,
            versions: Arc::new(Mutex::new(BTreeMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }
}

impl<I, C, R> Semaphore<I, C, R>
where
    I: Ord + fmt::Debug,
    C: Collate + Clone + Send + Sync,
    R: OverlapsRange<R, C> + fmt::Debug + Send + Sync,
{
    /// Construct a semaphore with a historical write reservation for its initial value.
    pub fn with_reservation(txn_id: I, collator: C, range: R) -> Self {
        let versions = BTreeMap::from([(txn_id, vec![Reservation::new(Arc::new(range), true)])]);
        Self {
            collator,
            versions: Arc::new(Mutex::new(versions)),
            notify: Arc::new(Notify::new()),
        }
    }

    fn attempt(&self, txn_id: I, range: Arc<R>, write: bool) -> Result<Attempt<I, R>> {
        let mut versions = self.versions.lock().expect("versions");

        // Historical rejection must not register an attempted reservation.
        for (id, reservations) in versions.iter() {
            match txn_id.cmp(id) {
                Ordering::Greater => {
                    if reservations
                        .iter()
                        .any(|r| r.write && r.range.contains_partial(&range, &self.collator))
                    {
                        return Ok(Attempt::Pending(txn_id, range));
                    }
                }
                Ordering::Less
                    if write
                        && reservations
                            .iter()
                            .any(|r| r.range.contains_partial(&range, &self.collator)) =>
                {
                    return Err(Error::Conflict);
                }
                _ => {}
            }
        }

        let count = if write { PERMITS } else { 1 };
        let owned = if let Some(reservations) = versions.get_mut(&txn_id) {
            let position = if let Some(position) = reservations
                .iter()
                .position(|r| r.range.overlaps(&range, &self.collator) == Overlap::Equal)
            {
                reservations[position].write |= write;
                position
            } else {
                reservations.push(Reservation::new(range.clone(), write));
                reservations.len() - 1
            };

            // An active permit protects its exact range, including reservations
            // added after it was acquired. Topology plays no role in exclusion.
            if reservations.iter().any(|r| {
                let available = r.semaphore.available_permits();
                r.range.contains_partial(&range, &self.collator)
                    && if write {
                        available < PERMITS as usize
                    } else {
                        available == 0
                    }
            }) {
                return Ok(Attempt::Pending(txn_id, range));
            }

            reservations[position]
                .semaphore
                .clone()
                .try_acquire_many_owned(count)?
        } else {
            let reservation = Reservation::new(range.clone(), write);
            let owned = reservation
                .semaphore
                .clone()
                .try_acquire_many_owned(count)?;
            versions.insert(txn_id, vec![reservation]);
            owned
        };

        Ok(Attempt::Acquired(Permit {
            range,
            owned: Some(owned),
            notify: self.notify.clone(),
        }))
    }

    async fn acquire(&self, mut txn_id: I, range: R, write: bool) -> Result<Permit<R>> {
        let mut range = Arc::new(range);
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            match self.attempt(txn_id, range, write)? {
                Attempt::Acquired(permit) => return Ok(permit),
                Attempt::Pending(id, requested) => {
                    txn_id = id;
                    range = requested;
                }
            }

            #[cfg(test)]
            tests::before_wait();
            notified.await;
        }
    }

    fn try_acquire(&self, txn_id: I, range: R, write: bool) -> Result<Permit<R>> {
        match self.attempt(txn_id, Arc::new(range), write)? {
            Attempt::Acquired(permit) => Ok(permit),
            Attempt::Pending(_, _) => Err(Error::WouldBlock),
        }
    }

    /// Acquire a permit to read a section of a transactional resource.
    pub async fn read(&self, txn_id: I, range: R) -> Result<PermitRead<R>> {
        self.acquire(txn_id, range, false)
            .await
            .map(|permit| PermitRead { permit })
    }

    /// Synchronously acquire a read permit, if possible.
    pub fn try_read(&self, txn_id: I, range: R) -> Result<PermitRead<R>> {
        self.try_acquire(txn_id, range, false)
            .map(|permit| PermitRead { permit })
    }

    /// Acquire a permit to write to a section of a transactional resource.
    pub async fn write(&self, txn_id: I, range: R) -> Result<PermitWrite<R>> {
        self.acquire(txn_id, range, true)
            .await
            .map(|permit| PermitWrite { permit })
    }

    /// Synchronously acquire a write permit, if possible.
    pub fn try_write(&self, txn_id: I, range: R) -> Result<PermitWrite<R>> {
        self.try_acquire(txn_id, range, true)
            .map(|permit| PermitWrite { permit })
    }
}

impl<I: Copy + Ord + fmt::Debug, C, R> Semaphore<I, C, R> {
    /// Release reservations after the caller has quiesced the affected operations.
    ///
    /// With `drop_past`, also remove all reservations earlier than `txn_id`.
    /// The caller owns lifecycle decisions; this semaphore stores no finalized frontier.
    pub fn finalize(&self, txn_id: &I, drop_past: bool) {
        let mut versions = self.versions.lock().expect("versions");
        let notify = if drop_past {
            let count = versions.len();
            while versions
                .first_key_value()
                .is_some_and(|(id, _)| id <= txn_id)
            {
                versions.pop_first();
            }
            versions.len() != count
        } else {
            versions.remove(txn_id).is_some()
        };
        drop(versions);

        if notify {
            self.notify.notify_waiters();
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/semaphore.rs"]
mod tests;
