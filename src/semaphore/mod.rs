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

mod version;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use collate::{Collate, OverlapsRange};
use tokio::sync::Notify;

use super::{Error, Result};

use version::{Permit as VersionPermit, Version};

/// A permit to read a specific section of a transactional resource
#[derive(Debug)]
pub struct PermitRead<R> {
    permit: VersionPermit<R>,
    _notify: NotifyOnDrop,
}

impl<R> Deref for PermitRead<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.permit.deref()
    }
}

/// A permit to write to a specific section of a transactional resource
#[derive(Debug)]
pub struct PermitWrite<R> {
    permit: VersionPermit<R>,
    _notify: NotifyOnDrop,
}

impl<R> Deref for PermitWrite<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.permit.deref()
    }
}

// Fields drop in declaration order: release the permit before waking callers.
#[derive(Debug)]
struct NotifyOnDrop(Arc<Notify>);

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        self.0.notify_waiters();
    }
}

enum Attempt<I, R> {
    Acquired(VersionPermit<R>),
    Pending(I, Arc<R>),
}

/// A semaphore used to maintain the ACID compliance of transactional resource
pub struct Semaphore<I, C, R> {
    collator: C,
    versions: Arc<Mutex<BTreeMap<I, Version<R>>>>,
    notify: Arc<Notify>,
}

impl<I, C, R> Clone for Semaphore<I, C, R>
where
    C: Clone,
{
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
    /// Construct a new transactional [`Semaphore`] with a write reservation for its initial value.
    pub fn with_reservation(txn_id: I, collator: C, range: R) -> Self {
        let mut version = Version::new();
        version.reserve(Arc::new(range), true, &collator);
        let mut versions = BTreeMap::new();
        versions.insert(txn_id, version);
        Self {
            collator,
            versions: Arc::new(Mutex::new(versions)),
            notify: Arc::new(Notify::new()),
        }
    }

    // No range description or partial permit escapes this atomic attempt.
    fn try_acquire(&self, txn_id: I, range: Arc<R>, write: bool) -> Result<Attempt<I, R>> {
        let mut versions = self.versions.lock().expect("versions");
        for (version_id, version) in versions.iter() {
            match txn_id.cmp(version_id) {
                Ordering::Greater => {
                    if version.is_pending_write_at(&range, &self.collator) {
                        return Ok(Attempt::Pending(txn_id, range));
                    }
                }
                Ordering::Less if write && version.has_been_read_at(&range, &self.collator) => {
                    return Err(Error::Conflict);
                }
                _ => {}
            }
        }

        if let Some(version) = versions.get_mut(&txn_id) {
            match version.try_acquire(range.clone(), write, &self.collator) {
                Ok(permit) => Ok(Attempt::Acquired(permit)),
                Err(Error::WouldBlock) => Ok(Attempt::Pending(txn_id, range)),
                Err(cause) => Err(cause),
            }
        } else {
            // A fresh transaction has no active permits. Keep I owned until this
            // succeeds so retries need no Clone or Copy bound on transaction IDs.
            let mut version = Version::new();
            let permit = version.try_acquire(range, write, &self.collator)?;
            versions.insert(txn_id, version);
            Ok(Attempt::Acquired(permit))
        }
    }

    async fn acquire(
        &self,
        mut txn_id: I,
        mut range: Arc<R>,
        write: bool,
    ) -> Result<VersionPermit<R>> {
        loop {
            // notify_waiters records notifications for this future before its
            // first poll. Register before checking both history and live permits.
            let notified = self.notify.notified();
            match self.try_acquire(txn_id, range, write)? {
                Attempt::Acquired(permit) => return Ok(permit),
                Attempt::Pending(id, pending_range) => {
                    txn_id = id;
                    range = pending_range;
                }
            }
            #[cfg(test)]
            tests::before_wait();
            notified.await;
        }
    }

    /// Acquire a permit to read a section of a transactional resource.
    /// Waiting is cancel-safe; wake-up order is not FIFO.
    pub async fn read(&self, txn_id: I, range: R) -> Result<PermitRead<R>> {
        let permit = self.acquire(txn_id, Arc::new(range), false).await?;
        Ok(PermitRead {
            permit,
            _notify: NotifyOnDrop(self.notify.clone()),
        })
    }

    /// Synchronously acquire a permit to read a section of a transactional resource, if possible.
    pub fn try_read(&self, txn_id: I, range: R) -> Result<PermitRead<R>> {
        match self.try_acquire(txn_id, Arc::new(range), false)? {
            Attempt::Acquired(permit) => Ok(PermitRead {
                permit,
                _notify: NotifyOnDrop(self.notify.clone()),
            }),
            Attempt::Pending(_, _) => Err(Error::WouldBlock),
        }
    }

    /// Acquire a permit to write to a section of a transactional resource.
    /// Waiting is cancel-safe; wake-up order is not FIFO.
    pub async fn write(&self, txn_id: I, range: R) -> Result<PermitWrite<R>> {
        let permit = self.acquire(txn_id, Arc::new(range), true).await?;
        Ok(PermitWrite {
            permit,
            _notify: NotifyOnDrop(self.notify.clone()),
        })
    }

    /// Synchronously acquire a permit to write to a section of a transactional resource, if possible.
    pub fn try_write(&self, txn_id: I, range: R) -> Result<PermitWrite<R>> {
        match self.try_acquire(txn_id, Arc::new(range), true)? {
            Attempt::Acquired(permit) => Ok(PermitWrite {
                permit,
                _notify: NotifyOnDrop(self.notify.clone()),
            }),
            Attempt::Pending(_, _) => Err(Error::WouldBlock),
        }
    }
}

impl<I: Copy + Ord + fmt::Debug, C, R> Semaphore<I, C, R> {
    /// Mark a transaction completed, un-blocking waiting requests for future permits.
    ///
    /// Call this when the transactional resource is no longer writable at `txn_id`
    /// (due to a commit, rollback, timeout, or any other reason).
    ///
    /// Set `drop_past` to `true` to finalize all transactions older than `txn_id`.
    pub fn finalize(&self, txn_id: &I, drop_past: bool) {
        let mut versions = self.versions.lock().expect("versions");

        let notify = if drop_past {
            let mut notify = false;

            while let Some(version_id) = versions.keys().next().copied() {
                if &version_id > txn_id {
                    break;
                } else {
                    assert!(versions.remove(&version_id).is_some());
                    notify = true;
                }
            }

            notify
        } else {
            versions.remove(txn_id).is_some()
        };

        drop(versions);
        if notify {
            #[cfg(feature = "logging")]
            log::trace!("dropped transactional semaphore version {txn_id:?}");

            self.notify.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set::Range;
    use collate::Collator;
    use std::cell::RefCell;

    tokio::task_local! {
        static BEFORE_WAIT: RefCell<Option<Box<dyn FnOnce()>>>;
    }

    pub(super) fn before_wait() {
        let _ = BEFORE_WAIT.try_with(|hook| {
            if let Some(hook) = hook.take() {
                hook();
            }
        });
    }

    #[test]
    fn initial_reservation_blocks_overlapping_later_access() {
        let semaphore = Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::with_reservation(
            1,
            Collator::default(),
            2..4,
        );
        assert!(matches!(
            semaphore.try_read(2, 3..4),
            Err(Error::WouldBlock)
        ));
        assert!(matches!(
            semaphore.try_write(2, 3..4),
            Err(Error::WouldBlock)
        ));
        assert!(semaphore.try_read(2, 4..5).is_ok());
        assert!(semaphore.try_write(2, 5..6).is_ok());
        semaphore.finalize(&1, false);
        assert!(semaphore.try_read(2, 3..4).is_ok());
        assert!(semaphore.try_write(2, 3..4).is_ok());
    }

    #[tokio::test]
    async fn finalization_between_reservation_check_and_wait_is_observed() {
        for write in [false, true] {
            let semaphore = Semaphore::<u64, Collator<u64>, Range<u64>>::new(Collator::default());
            drop(semaphore.try_write(1, Range::All).unwrap());
            let blocker = semaphore.clone();
            BEFORE_WAIT
                .scope(
                    RefCell::new(Some(Box::new(move || blocker.finalize(&1, false)))),
                    async {
                        let waiting = async {
                            if write {
                                drop(semaphore.write(2, Range::All).await.unwrap());
                            } else {
                                drop(semaphore.read(2, Range::All).await.unwrap());
                            }
                        };
                        futures::pin_mut!(waiting);
                        assert!(futures::poll!(waiting).is_ready());
                        assert!(BEFORE_WAIT.with(|hook| hook.borrow().is_none()));
                    },
                )
                .await;
        }
    }

    #[tokio::test]
    async fn cancelled_waiters_leave_no_later_reservation() {
        for write in [false, true] {
            let semaphore = Semaphore::<u64, Collator<u64>, Range<u64>>::new(Collator::default());
            drop(semaphore.try_write(1, Range::All).unwrap());
            {
                let waiting = async {
                    if write {
                        drop(semaphore.write(2, Range::All).await.unwrap());
                    } else {
                        drop(semaphore.read(2, Range::All).await.unwrap());
                    }
                };
                futures::pin_mut!(waiting);
                assert!(futures::poll!(waiting).is_pending());
            }
            assert_eq!(semaphore.versions.lock().unwrap().len(), 1);
            semaphore.finalize(&1, false);
            assert!(semaphore.versions.lock().unwrap().is_empty());
            assert!(semaphore.try_write(3, Range::All).is_ok());
        }
    }

    #[tokio::test]
    async fn covering_read_blocks_a_later_inserted_child_write() {
        let semaphore = Semaphore::<u64, Collator<u64>, Range<u64>>::new(Collator::default());
        let read = semaphore.try_read(1, Range::All).unwrap();
        assert!(matches!(
            semaphore.try_write(1, Range::One(Arc::new(2))),
            Err(Error::WouldBlock)
        ));
        assert!(tokio::time::timeout(
            std::time::Duration::from_millis(10),
            semaphore.write(1, Range::One(Arc::new(2)))
        )
        .await
        .is_err());
        drop(read);
        let write = semaphore.try_write(1, Range::One(Arc::new(2))).unwrap();
        assert!(matches!(
            semaphore.try_read(1, Range::All),
            Err(Error::WouldBlock)
        ));
        drop(write);
        assert!(semaphore.try_read(1, Range::All).is_ok());
    }

    #[test]
    fn exact_ranges_preserve_overlap_exclusion_and_disjoint_independence() {
        let cases = [
            (4..8, 4..8, true),
            (0..20, 5..10, true),
            (5..10, 0..20, true),
            (0..10, 5..15, true),
            (5..15, 0..10, true),
            (0..5, 5..10, false),
            (10..20, 70..80, false),
            (70..80, 10..20, false),
        ];
        for (first, second, overlaps) in cases {
            for first_write in [false, true] {
                for second_write in [false, true] {
                    let semaphore = Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(
                        Collator::default(),
                    );
                    // Historical covering and intermediate ranges must not make
                    // disjoint live operations conflict with each other.
                    for history in [0..100, 40..60, 20..80] {
                        drop(semaphore.try_read(1, history).unwrap());
                    }
                    let held: Box<dyn std::fmt::Debug> = if first_write {
                        Box::new(semaphore.try_write(1, first.clone()).unwrap())
                    } else {
                        Box::new(semaphore.try_read(1, first.clone()).unwrap())
                    };
                    let result = if second_write {
                        semaphore.try_write(1, second.clone()).map(drop)
                    } else {
                        semaphore.try_read(1, second.clone()).map(drop)
                    };
                    assert_eq!(
                        result,
                        if overlaps && (first_write || second_write) {
                            Err(Error::WouldBlock)
                        } else {
                            Ok(())
                        },
                        "{first:?} write={first_write}, {second:?} write={second_write}"
                    );
                    drop(held);
                    assert!(semaphore.try_write(1, second.clone()).is_ok());
                }
            }
        }
    }

    #[test]
    fn ordering_checks_exact_history_including_partial_tails() {
        let semaphore =
            Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
        drop(semaphore.try_read(1, 0..100).unwrap());
        drop(semaphore.try_write(1, 10..20).unwrap());
        assert!(semaphore.try_read(2, 70..80).is_ok());
        assert!(matches!(
            semaphore.try_read(2, 15..16),
            Err(Error::WouldBlock)
        ));
        semaphore.finalize(&1, false);
        assert!(semaphore.try_read(2, 15..16).is_ok());

        let semaphore =
            Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
        for range in [0..5, 10..15, 3..12] {
            drop(semaphore.try_read(3, range).unwrap());
        }
        for tail in [0..1, 14..15] {
            assert!(matches!(semaphore.try_write(2, tail), Err(Error::Conflict)));
            assert!(!semaphore.versions.lock().unwrap().contains_key(&2));
        }
        assert!(semaphore.try_write(2, 15..16).is_ok());
    }

    #[tokio::test]
    async fn blocked_same_transaction_attempts_retain_history_without_active_permits() {
        for asynchronous in [false, true] {
            let semaphore =
                Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
            let read = semaphore.try_read(1, 0..5).unwrap();
            if asynchronous {
                let waiting = semaphore.write(1, 0..20);
                futures::pin_mut!(waiting);
                assert!(futures::poll!(waiting).is_pending());
                // Dropping the future cancels waiting, not its range history.
            } else {
                assert!(matches!(
                    semaphore.try_write(1, 0..20),
                    Err(Error::WouldBlock)
                ));
            }
            drop(read);
            assert!(matches!(
                semaphore.try_read(2, 10..15),
                Err(Error::WouldBlock)
            ));
            assert!(semaphore.try_write(1, 10..15).is_ok());
            semaphore.finalize(&1, false);
            assert!(semaphore.try_read(2, 10..15).is_ok());
        }
    }

    #[tokio::test]
    async fn permit_release_between_attempt_and_wait_is_observed() {
        for write in [false, true] {
            let semaphore = Semaphore::<u64, Collator<u64>, Range<u64>>::new(Collator::default());
            let held = semaphore.try_write(1, Range::All).unwrap();
            BEFORE_WAIT
                .scope(RefCell::new(Some(Box::new(move || drop(held)))), async {
                    let waiting = async {
                        if write {
                            drop(semaphore.write(1, Range::All).await.unwrap());
                        } else {
                            drop(semaphore.read(1, Range::All).await.unwrap());
                        }
                    };
                    futures::pin_mut!(waiting);
                    assert!(futures::poll!(waiting).is_ready());
                    assert!(BEFORE_WAIT.with(|hook| hook.borrow().is_none()));
                })
                .await;
        }
    }

    #[tokio::test]
    async fn notification_observes_released_permits_and_unlocked_history() {
        use futures::task::{waker, ArcWake};
        use std::future::Future;
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::task::Context;

        struct Observer {
            semaphore: Semaphore<u64, Collator<u64>, Range<u64>>,
            txn_id: u64,
            ready: AtomicBool,
        }
        impl ArcWake for Observer {
            fn wake_by_ref(this: &Arc<Self>) {
                // A custom waker may run immediately inside notify_waiters.
                // Never hang the regression if it observes the old lock order.
                let unlocked = this.semaphore.versions.try_lock().is_ok();
                let ready = unlocked && this.semaphore.try_write(this.txn_id, Range::All).is_ok();
                this.ready.store(ready, AtomicOrdering::SeqCst);
            }
        }

        for finalize in [false, true] {
            let semaphore = Semaphore::<u64, Collator<u64>, Range<u64>>::new(Collator::default());
            let held = semaphore.try_write(1, Range::All).unwrap();
            let txn_id = if finalize { 2 } else { 1 };
            let observer = Arc::new(Observer {
                semaphore: semaphore.clone(),
                txn_id,
                ready: AtomicBool::new(false),
            });
            let wake = waker(observer.clone());
            let mut context = Context::from_waker(&wake);
            let waiting = semaphore.read(txn_id, Range::All);
            futures::pin_mut!(waiting);
            assert!(waiting.as_mut().poll(&mut context).is_pending());
            if finalize {
                drop(held);
                // Re-poll after permit release: history still blocks txn 2.
                assert!(waiting.as_mut().poll(&mut context).is_pending());
                observer.ready.store(false, AtomicOrdering::SeqCst);
                semaphore.finalize(&1, false);
            } else {
                drop(held);
            }
            assert!(observer.ready.load(AtomicOrdering::SeqCst));
            assert!(waiting.as_mut().poll(&mut context).is_ready());
        }
    }

    #[tokio::test]
    async fn acquisition_requires_no_clone_bound_on_ids_or_ranges() {
        #[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
        struct Id(u64);
        #[derive(Debug)]
        struct Selection(std::ops::Range<u64>);
        impl OverlapsRange<Self, Collator<u64>> for Selection {
            fn overlaps(&self, other: &Self, collator: &Collator<u64>) -> collate::Overlap {
                self.0.overlaps(&other.0, collator)
            }
        }

        let semaphore = Semaphore::<Id, Collator<u64>, Selection>::new(Collator::default());
        let read = semaphore.try_read(Id(1), Selection(2..4)).unwrap();
        let waiting = semaphore.write(Id(1), Selection(2..4));
        futures::pin_mut!(waiting);
        assert!(futures::poll!(waiting.as_mut()).is_pending());
        drop(read);
        let write = waiting.await.unwrap();
        assert_eq!(write.0, 2..4);
    }
}
