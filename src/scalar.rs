//! A transactional lock on a scalar value.
//!
//! Example:
//! ```
//! use futures::executor::block_on;
//! use txn_lock::scalar::*;
//! use txn_lock::Error;
//!
//! let lock = TxnLock::new("zero");
//!
//! let value = block_on(lock.read(1)).expect("read");
//! assert_eq!(*value, "zero");
//! assert_eq!(lock.try_write(1).unwrap_err(), Error::WouldBlock);
//! assert_eq!(*lock.try_read(1).expect("read"), "zero");
//!
//! lock.commit(1).unwrap();
//!
//! {
//!     let mut guard = lock.try_write(2).expect("write lock");
//!     *guard = "two";
//! }
//!
//! assert_eq!(*lock.try_read(0).expect("read past version"), "zero");
//! assert_eq!(*lock.try_read(2).expect("read current version"), "two");
//!
//! lock.commit(2).unwrap();
//!
//! assert_eq!(*lock.try_read(3).expect("new value"), "two");
//!
//! lock.rollback(&3).unwrap();
//!
//! {
//!     let mut guard = lock.try_write(3).expect("write lock");
//!     *guard = "three";
//! }
//!
//! lock.finalize(1).unwrap();
//!
//! assert_eq!(lock.try_read(0).unwrap_err(), Error::Outdated);
//! assert_eq!(*lock.try_read(3).expect("current value"), "three");
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock as RwLockInner};
use std::task::Poll;

use collate::{Collator, Overlap, OverlapsRange};
use tokio::sync::RwLock;

use super::guard::{TxnReadGuard, TxnWriteGuard};
use super::semaphore::{PermitRead, PermitWrite, Semaphore};
use super::{Error, Result};

/// A read guard on a [`TxnLock`]
pub type TxnLockReadGuard<T> = TxnReadGuard<Range, T>;

/// A write guard on a [`TxnLock`]
pub type TxnLockWriteGuard<T> = TxnWriteGuard<Range, T>;

/// A range used to reserve a permit to guard access to a [`TxnLock`]
#[derive(Debug)]
pub struct Range;

impl OverlapsRange<Range, Collator<()>> for Range {
    fn overlaps(&self, _other: &Range, _collator: &Collator<()>) -> Overlap {
        Overlap::Equal
    }
}

struct State<I, T> {
    canon: Arc<T>,
    deltas: BTreeMap<I, Arc<T>>,
    commits: BTreeSet<I>,
    pending: BTreeMap<I, Arc<RwLock<T>>>,
    finalized: Option<I>,
}

impl<I, T> State<I, T> {
    #[inline]
    fn new(canon: T) -> Self {
        State {
            canon: Arc::new(canon),
            deltas: BTreeMap::new(),
            commits: BTreeSet::new(),
            pending: BTreeMap::new(),
            finalized: None,
        }
    }
}

impl<I: Ord + Hash + Copy + fmt::Debug, T: fmt::Debug> State<I, T> {
    #[inline]
    fn check_pending(&self, txn_id: &I) -> Result<()> {
        if self.finalized.as_ref() >= Some(txn_id) {
            Err(Error::Outdated)
        } else if self.commits.contains(txn_id) {
            Err(Error::Committed)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn commit_version(&mut self, txn_id: &I) -> Result<Option<Arc<T>>> {
        if self
            .pending
            .get(txn_id)
            .is_some_and(|value| Arc::strong_count(value) != 1)
        {
            return Err(Error::Busy);
        }

        // The state lock prevents new references after the uniqueness check.
        Ok(self.pending.remove(txn_id).map(|version| {
            Arc::new(
                Arc::try_unwrap(version)
                    .expect("exclusive pending value")
                    .into_inner(),
            )
        }))
    }

    #[inline]
    fn commit(&mut self, txn_id: I) -> Result<()> {
        if self.commits.contains(&txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {:?}", txn_id);
        } else if let Some(new_value) = self.commit_version(&txn_id)? {
            assert!(self.deltas.insert(txn_id, new_value).is_none());
        }

        self.commits.insert(txn_id);
        Ok(())
    }

    #[inline]
    fn finalize(&mut self, txn_id: I) -> Option<&Arc<T>> {
        if self.finalized > Some(txn_id) {
            return None;
        }

        while let Some(entry) = self.pending.first_entry() {
            if entry.key() > &txn_id {
                break;
            }

            entry.remove_entry();
        }

        while self.commits.first().is_some_and(|id| id <= &txn_id) {
            self.commits.pop_first();
        }

        while let Some(entry) = self.deltas.first_entry() {
            if entry.key() > &txn_id {
                break;
            }

            let (_, version) = entry.remove_entry();
            self.canon = version;
        }

        self.finalized = Some(txn_id);

        Some(&self.canon)
    }

    #[inline]
    fn read_canon(&self, txn_id: &I) -> &Arc<T> {
        self.deltas
            .range(..=txn_id)
            .next_back()
            .map(|(_, version)| version)
            .unwrap_or(&self.canon)
    }

    #[inline]
    fn read_committed(&self, txn_id: &I) -> Poll<Result<Arc<T>>> {
        if self.finalized.as_ref() > Some(txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.finalized.as_ref() == Some(txn_id) || self.commits.contains(txn_id) {
            debug_assert!(!self.pending.contains_key(txn_id));
            Poll::Ready(Ok(self.read_canon(txn_id).clone()))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn read_pending(&mut self, txn_id: &I, permit: PermitRead<Range>) -> TxnLockReadGuard<T> {
        if let Some(version) = self.pending.get(txn_id) {
            // the permit means it's safe to call try_read_owned().expect()
            let value = version.clone().try_read_owned().expect("version");
            TxnLockReadGuard::pending_write(permit, value)
        } else {
            let value = self.read_canon(txn_id).clone();
            TxnLockReadGuard::pending_read(permit, value)
        }
    }
}

impl<I: Copy + Ord + Hash + fmt::Debug, T: Clone + fmt::Debug> State<I, T> {
    #[inline]
    fn get_or_create_version(&mut self, txn_id: I) -> Arc<RwLock<T>> {
        if let Some(version) = self.pending.get(&txn_id) {
            version.clone()
        } else {
            let canon = self.read_canon(&txn_id);
            let value = T::clone(canon);
            let version = Arc::new(RwLock::new(value));
            self.pending.insert(txn_id, version.clone());
            version
        }
    }

    #[inline]
    fn write(&mut self, txn_id: I, permit: PermitWrite<Range>) -> Result<TxnLockWriteGuard<T>> {
        // the permit means it's safe to call try_write_owned().expect()
        let value = self
            .get_or_create_version(txn_id)
            .try_write_owned()
            .expect("version");

        Ok(TxnLockWriteGuard::new(permit, value))
    }
}

/// A futures-aware read-write lock on a scalar value which supports transactional versioning.
///
/// For non-scalar data types (e.g. a list, set, map, etc) consider using an alternate lock type,
/// or implementing a custom lock type using [`Semaphore`].
///
/// The type `T` to lock must implement [`Clone`] in order to support versioning.
/// [`Clone::clone`] is called once when [`TxnLock::write`] is called with a valid new `txn_id`.
pub struct TxnLock<I, T> {
    state: Arc<RwLockInner<State<I, T>>>,
    semaphore: Semaphore<I, Collator<()>, Range>,
}

impl<I, T> Clone for TxnLock<I, T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I, T> TxnLock<I, T> {
    #[inline]
    fn state(&self) -> impl Deref<Target = State<I, T>> + '_ {
        self.state.read().expect("transaction bookkeeping poisoned")
    }

    #[inline]
    fn state_mut(&self) -> impl DerefMut<Target = State<I, T>> + '_ {
        self.state
            .write()
            .expect("transaction bookkeeping poisoned")
    }
}

impl<I, T> TxnLock<I, T> {
    /// Return whether both handles refer to the same transactional owner.
    /// This compares ownership identity, not stored values or transaction state.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Construct a new [`TxnLock`].
    pub fn new(canon: T) -> Self {
        Self {
            state: Arc::new(RwLockInner::new(State::new(canon))),
            semaphore: Semaphore::new(Collator::default()),
        }
    }
}

impl<I, T> TxnLock<I, T>
where
    I: Copy + Hash + Ord + fmt::Debug,
    T: fmt::Debug,
{
    /// Inspect the baseline, latest committed value, and finalized frontier at `txn_id`.
    ///
    /// This is metadata inspection, not a replacement for transactional reads.
    /// The caller must retain its transactional guard and synchronize any external
    /// publication with this inspection. The callback runs under the bookkeeping
    /// lock and must not block or acquire other locks.
    pub fn inspect_committed<R>(
        &self,
        txn_id: I,
        inspect: impl FnOnce(&T, &T, Option<I>) -> R,
    ) -> Result<R> {
        let state = self.state();
        if state.finalized > Some(txn_id) {
            return Err(Error::Outdated);
        }

        let latest = state
            .deltas
            .last_key_value()
            .map_or(&state.canon, |(_, value)| value);

        Ok(inspect(&state.canon, latest, state.finalized))
    }

    /// Acquire a read lock for this value at `txn_id`.
    pub async fn read(&self, txn_id: I) -> Result<TxnLockReadGuard<T>> {
        if let Poll::Ready(result) = self.state().read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        let permit = self.semaphore.read(txn_id, Range).await?;

        let mut state = self.state_mut();

        if let Poll::Ready(result) = state.read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        Ok(state.read_pending(&txn_id, permit))
    }

    /// Acquire a read lock for this value at `txn_id` synchronously, if possible.
    pub fn try_read(&self, txn_id: I) -> Result<TxnLockReadGuard<T>> {
        let mut state = self.state_mut();

        if let Poll::Ready(result) = state.read_committed(&txn_id) {
            return result.map(TxnLockReadGuard::Committed);
        }

        let permit = self.semaphore.try_read(txn_id, Range)?;
        Ok(state.read_pending(&txn_id, permit))
    }
}

impl<I, T> TxnLock<I, T>
where
    I: Copy + Hash + Ord + fmt::Debug,
    T: Clone + fmt::Debug,
{
    /// Acquire a write lock for this value at `txn_id`.
    pub async fn write(&self, txn_id: I) -> Result<TxnLockWriteGuard<T>> {
        self.state().check_pending(&txn_id)?;

        let permit = self.semaphore.write(txn_id, Range).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;
        state.write(txn_id, permit)
    }

    /// Acquire a write lock for this value at `txn_id` synchronously, if possible.
    pub fn try_write(&self, txn_id: I) -> Result<TxnLockWriteGuard<T>> {
        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let permit = self.semaphore.try_write(txn_id, Range)?;
        state.write(txn_id, permit)
    }
}

impl<I, T> TxnLock<I, T>
where
    I: Copy + Hash + Ord + fmt::Debug,
    T: fmt::Debug,
{
    /// Commit the state of this [`TxnLock`] at `txn_id`.
    /// Returns an error for an invalid decision or unavailable state.
    pub fn commit(&self, txn_id: I) -> Result<()> {
        let mut state = self.state_mut();
        if state.finalized == Some(txn_id) {
            return Ok(());
        }

        if state.finalized > Some(txn_id) {
            return Err(Error::Outdated);
        } else {
            state.commit(txn_id)?;
        }

        self.semaphore.finalize(&txn_id, false)?;
        Ok(())
    }

    /// Read and commit this [`TxnLock`] in a single operation.
    /// Returns an error for an invalid decision or unavailable state.
    pub async fn read_and_commit(&self, txn_id: I) -> Result<Arc<T>> {
        // Reject a known invalid decision before registering a reservation.
        {
            let state = self.state();
            if state.finalized == Some(txn_id) {
                return Ok(state.read_canon(&txn_id).clone());
            }

            if state.finalized > Some(txn_id) {
                return Err(Error::Outdated);
            }
        }

        let _permit = self.semaphore.read(txn_id, Range).await?;

        let canon = {
            let mut state = self.state_mut();
            if state.finalized == Some(txn_id) {
                return Ok(state.read_canon(&txn_id).clone());
            }

            if state.finalized > Some(txn_id) {
                return Err(Error::Outdated);
            } else {
                state.commit(txn_id)?;
                state.read_canon(&txn_id).clone()
            }
        };

        self.semaphore.finalize(&txn_id, false)?;

        Ok(canon)
    }

    /// Roll back the state of this [`TxnLock`] at `txn_id`.
    /// Returns an error for an invalid decision or unavailable state.
    pub fn rollback(&self, txn_id: &I) -> Result<()> {
        let mut state = self.state_mut();
        if state.finalized.as_ref() == Some(txn_id) {
            return Ok(());
        }

        if state.commits.contains(txn_id) {
            return Err(Error::Committed);
        }

        if state.finalized.as_ref() > Some(txn_id) {
            return Err(Error::Outdated);
        }

        state.pending.remove(txn_id);

        self.semaphore.finalize(txn_id, false)?;
        Ok(())
    }

    /// Read and roll back this [`TxnLock`] in a single operation, if there is a version pending.
    /// Returns an error for an invalid decision or unavailable state.
    pub async fn read_and_rollback(&self, txn_id: I) -> Result<Arc<T>> {
        // Reject a known invalid decision before registering a reservation.
        {
            let state = self.state();
            if state.finalized == Some(txn_id) {
                return Ok(state.read_canon(&txn_id).clone());
            }

            if state.commits.contains(&txn_id) {
                return Err(Error::Committed);
            }

            if state.finalized > Some(txn_id) {
                return Err(Error::Outdated);
            }
        }

        let _permit = self.semaphore.read(txn_id, Range).await?;

        let version = {
            let mut state = self.state_mut();
            if state.finalized == Some(txn_id) {
                return Ok(state.read_canon(&txn_id).clone());
            }

            if state.commits.contains(&txn_id) {
                return Err(Error::Committed);
            }

            if state.finalized > Some(txn_id) {
                return Err(Error::Outdated);
            }

            state
                .commit_version(&txn_id)?
                .unwrap_or_else(|| state.read_canon(&txn_id).clone())
        };

        self.semaphore.finalize(&txn_id, false)?;

        Ok(version)
    }

    /// Finalize the state of this [`TxnLock`] at `txn_id`.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) -> Result<()> {
        let mut state = self.state_mut();
        self.semaphore.finalize(&txn_id, true)?;
        state.finalize(txn_id);
        Ok(())
    }

    /// Read and finalize the state of this [`TxnLock`] at `txn_id`, if not already finalized.
    /// This will merge in deltas and prevent further reads of versions earlier than `txn_id`.
    pub fn read_and_finalize(&self, txn_id: I) -> Result<Option<Arc<T>>> {
        let mut state = self.state_mut();
        self.semaphore.finalize(&txn_id, true)?;
        Ok(state.finalize(txn_id).cloned())
    }
}

impl<I, T> fmt::Debug for TxnLock<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a transactional lock on a scalar")
    }
}
