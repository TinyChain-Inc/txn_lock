use std::ops::Deref;
use std::sync::Arc;

use collate::{Collate, Overlap, OverlapsRange};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{Error, Result};

const PERMITS: u32 = u32::MAX >> 3;

/// An active capability for exactly one registered range.
#[derive(Debug)]
pub(super) struct Permit<R> {
    range: Arc<R>,
    _permit: OwnedSemaphorePermit,
}

impl<R> Deref for Permit<R> {
    type Target = R;

    fn deref(&self) -> &R {
        &self.range
    }
}

struct Reservation<R> {
    range: Arc<R>,
    write: bool,
    semaphore: Arc<Semaphore>,
}

/// Exact historical reservations within one transaction. The containing
/// Semaphore's mutex serializes registration, conflict checks, and acquisition.
pub(super) struct Version<R> {
    reservations: Vec<Reservation<R>>,
}

impl<R> Version<R> {
    pub fn new() -> Self {
        Self {
            reservations: Vec::new(),
        }
    }

    pub fn reserve<C>(&mut self, range: Arc<R>, write: bool, collator: &C) -> usize
    where
        C: Collate,
        R: OverlapsRange<R, C>,
    {
        if let Some(index) = self
            .reservations
            .iter()
            .position(|reservation| reservation.range.overlaps(&range, collator) == Overlap::Equal)
        {
            self.reservations[index].write |= write;
            index
        } else {
            let index = self.reservations.len();
            self.reservations.push(Reservation {
                range,
                write,
                semaphore: Arc::new(Semaphore::new(PERMITS as usize)),
            });
            index
        }
    }

    pub fn try_acquire<C>(&mut self, range: Arc<R>, write: bool, collator: &C) -> Result<Permit<R>>
    where
        C: Collate,
        R: OverlapsRange<R, C>,
    {
        // A same-transaction attempt retains its history even if active permits
        // block acquisition. Cancellation must not erase that reservation.
        let index = self.reserve(range.clone(), write, collator);
        let blocked = self.reservations.iter().any(|reservation| {
            if !reservation.range.contains_partial(&range, collator) {
                return false;
            }
            let available = reservation.semaphore.available_permits();
            if write {
                available != PERMITS as usize
            } else {
                available == 0
            }
        });
        if blocked {
            return Err(Error::WouldBlock);
        }

        let reservation = &self.reservations[index];
        let permit = reservation
            .semaphore
            .clone()
            .try_acquire_many_owned(if write { PERMITS } else { 1 })?;
        Ok(Permit {
            range: reservation.range.clone(),
            _permit: permit,
        })
    }

    /// Any historical access conflicts with an earlier overlapping write.
    pub fn has_been_read_at<C>(&self, target: &R, collator: &C) -> bool
    where
        C: Collate,
        R: OverlapsRange<R, C>,
    {
        self.reservations
            .iter()
            .any(|reservation| reservation.range.contains_partial(target, collator))
    }

    pub fn is_pending_write_at<C>(&self, target: &R, collator: &C) -> bool
    where
        C: Collate,
        R: OverlapsRange<R, C>,
    {
        self.reservations.iter().any(|reservation| {
            reservation.write && reservation.range.contains_partial(target, collator)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use collate::Collator;

    #[test]
    fn equal_ranges_share_one_reservation_and_preserve_write_history() {
        let mut version = Version::new();
        let collator = Collator::<u64>::default();
        let read = version
            .try_acquire(Arc::new(2..4), false, &collator)
            .unwrap();
        for _ in 0..64 {
            drop(
                version
                    .try_acquire(Arc::new(2..4), false, &collator)
                    .unwrap(),
            );
            assert!(matches!(
                version.try_acquire(Arc::new(2..4), true, &collator),
                Err(Error::WouldBlock)
            ));
        }
        assert_eq!(version.reservations.len(), 1);
        assert!(version.reservations[0].write);
        drop(read);
        drop(
            version
                .try_acquire(Arc::new(2..4), true, &collator)
                .unwrap(),
        );
        assert!(version.is_pending_write_at(&(2..4), &collator));
    }
}
