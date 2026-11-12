//! A futures-aware read-write lock on a [`HashMap`] which supports transactional versioning.
//!
//! Example usage:
//! ```
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! use futures::executor::block_on;
//!
//! use txn_lock::map::*;
//! use txn_lock::Error;
//!
//! let one = "one";
//! let two = "two";
//!
//! let map = TxnMapLock::<u64, String, f32>::new(1);
//!
//! assert_eq!(block_on(map.insert(1, one.to_string(), 1.0)).expect("insert"), None);
//!
//! let value = block_on(map.get(1, one)).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//!
//! assert_eq!(map.try_insert(1, one.to_string(), 2.0).unwrap_err(), Error::WouldBlock);
//!
//! std::mem::drop(value);
//!
//! let (state_at_txn_1, deltas) = block_on(map.read_and_commit(1)).unwrap();
//! assert_eq!(deltas.expect("deltas").len(), 1);
//! assert_eq!(state_at_txn_1.len(), 1);
//!
//! let mut value = map.try_get_mut(2, one).expect("read").expect("value");
//! assert_eq!(value, 1.0);
//! *value = 2.0;
//!
//! assert_eq!(map.try_remove(2, one).unwrap_err(), Error::WouldBlock);
//! std::mem::drop(value);
//!
//! let value = block_on(map.remove(2, one)).expect("remove").expect("value");
//! assert_eq!(*value, 2.0);
//!
//! assert_eq!(map.try_insert(2, two.to_string(), 1.0).expect("insert"), None);
//!
//! assert!(map.try_remove(2, one).expect("remove").is_none());
//!
//! assert_eq!(map.try_insert(2, two.to_string(), 2.0).expect("insert"), Some(1.0.into()));
//!
//! map.rollback(&2).unwrap();
//!
//! let value = map.try_get(1, one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! assert!(map.try_remove(3, two).expect("remove").is_none());
//!
//! map.finalize(2).unwrap();
//!
//! assert_eq!(map.try_get(1, one).unwrap_err(), Error::Outdated);
//!
//! let value = map.try_get(3, one).expect("read");
//! assert_eq!(*(value.expect("guard")), 1.0);
//!
//! map.commit(3).unwrap();
//!
//! let extension = [
//!     ("one".to_string(), 1.0),
//!     ("two".to_string(), 2.0),
//!     ("three".to_string(), 3.0),
//!     ("four".to_string(), 4.0),
//!     ("five".to_string(), 5.0),
//! ];
//!
//! map.try_extend(4, extension).expect("extend");
//!
//! for (key, mut value) in map.try_iter_mut(4).expect("iter") {
//!     *value *= 2.;
//! }
//!
//! let expected = HashMap::<String, f32>::from_iter([
//!     ("two".to_string(), 4.0),
//!     ("three".to_string(), 6.0),
//!     ("one".to_string(), 2.0),
//!     ("four".to_string(), 8.0),
//!     ("five".to_string(), 10.0),
//! ]);
//!
//! let actual = map.try_iter(4).expect("iter").collect::<Vec<_>>();
//! assert_eq!(actual.len(), expected.len());
//!
//! for (k, v) in actual {
//!     assert_eq!(expected.get(&*k), Some(&*v));
//! }
//!
//! let actual = map.try_clear(4).expect("clear");
//! assert_eq!(actual, expected.into_iter().map(|(k, v)| (k.into(), v.into())).collect());
//!
//! ```

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::{self, HashMap};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock as RwLockInner};
use std::task::Poll;
use std::{fmt, iter};

use collate::Collator;
use futures::TryFutureExt;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use super::guard::{TxnReadGuard, TxnReadGuardMap, TxnWriteGuard};
pub use super::range::{Key, Range};
use super::semaphore::*;
use super::{Error, Result};

type Canon<K, V> = HashMap<Key<K>, Arc<V>>;
type CommitResult<K, V> = Result<(Canon<K, V>, Option<Delta<K, V>>)>;
type Delta<K, V> = HashMap<Key<K>, Option<Arc<V>>>;
type Deltas<I, K, V> = BTreeMap<I, Delta<K, V>>;
type Pending<K, V> = HashMap<Key<K>, Option<Arc<RwLock<V>>>>;

/// A read guard on a value in a [`TxnMapLock`]
pub type TxnMapValueReadGuard<K, V> = TxnReadGuard<Range<K>, V>;

/// A mapped read guard on a value in a [`TxnMapLock`]
pub type TxnMapValueReadGuardMap<K, V> = TxnReadGuardMap<Range<K>, V>;

/// A write guard on a value in a [`TxnMapLock`]
pub type TxnMapValueWriteGuard<K, V> = TxnWriteGuard<Range<K>, V>;

#[derive(Debug)]
enum PendingValue<V> {
    Committed(Arc<V>),
    Pending(OwnedRwLockReadGuard<V>),
}

struct State<I, K, V> {
    canon: Canon<K, V>,
    commits: BTreeSet<I>,
    deltas: Deltas<I, K, V>,
    pending: BTreeMap<I, Pending<K, V>>,
    finalized: Option<I>,
}

impl<I: Ord + Hash + fmt::Debug, K, V> State<I, K, V> {
    #[inline]
    fn new(txn_id: I, version: Pending<K, V>) -> Self {
        Self {
            canon: Canon::new(),
            commits: BTreeSet::new(),
            deltas: BTreeMap::new(),
            pending: BTreeMap::from_iter(iter::once((txn_id, version))),
            finalized: None,
        }
    }

    fn from_committed(canon: Canon<K, V>) -> Self {
        Self {
            canon,
            commits: BTreeSet::new(),
            deltas: BTreeMap::new(),
            pending: BTreeMap::new(),
            finalized: None,
        }
    }
}

impl<I, K, V> State<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: fmt::Debug,
{
    #[inline]
    fn check_committed(&self, txn_id: &I) -> Result<bool> {
        match self.finalized.as_ref().cmp(&Some(txn_id)) {
            Ordering::Greater => Err(Error::Outdated),
            Ordering::Equal => Ok(true),
            Ordering::Less => Ok(self.commits.contains(txn_id)),
        }
    }

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
    fn canon(&self, txn_id: &I) -> Canon<K, V> {
        let mut canon = self.canon.clone();

        let deltas = self.deltas.range(..=txn_id).map(|(_id, version)| version);

        for version in deltas {
            for (key, delta) in version {
                if let Some(value) = delta {
                    canon.insert(key.clone(), value.clone());
                } else {
                    canon.remove(key);
                }
            }
        }

        canon
    }

    #[inline]
    fn clear(&mut self, txn_id: I) -> Canon<K, V> {
        let mut map = self.canon(&txn_id);

        if let Some(version) = self.pending.remove(&txn_id) {
            for (key, delta) in version {
                if let Some(value) = delta {
                    let value = Arc::try_unwrap(value).expect("value");
                    map.insert(key, Arc::new(value.into_inner()));
                } else {
                    map.remove(&key);
                }
            }
        }

        let version = map.keys().cloned().map(|key| (key, None)).collect();
        self.pending.insert(txn_id, version);

        map
    }

    #[inline]
    fn check_commit(&self, txn_id: &I) -> Result<()> {
        if self.check_committed(txn_id)? {
            Ok(())
        } else {
            self.check_pending_values(txn_id)
        }
    }

    #[inline]
    fn check_pending_values(&self, txn_id: &I) -> Result<()> {
        if self.pending.get(txn_id).is_some_and(|version| {
            version
                .values()
                .flatten()
                .any(|value| Arc::strong_count(value) != 1)
        }) {
            return Err(Error::Busy);
        }

        Ok(())
    }

    #[inline]
    fn commit_version(&mut self, txn_id: &I) -> Result<Option<Delta<K, V>>> {
        self.check_pending_values(txn_id)?;

        // Check the entire version before removing anything. The state lock
        // prevents new references while these exclusively owned values move.
        Ok(self.pending.remove(txn_id).map(|version| {
            version
                .into_iter()
                .map(|(key, delta)| {
                    let value = delta.map(|value| {
                        Arc::new(
                            Arc::try_unwrap(value)
                                .expect("exclusive pending value")
                                .into_inner(),
                        )
                    });

                    (key, value)
                })
                .collect()
        }))
    }

    #[inline]
    fn commit(&mut self, txn_id: I) -> Result<()> {
        if self.commits.contains(&txn_id) {
            #[cfg(feature = "logging")]
            log::warn!("duplicate commit at {:?}", txn_id);
        } else if let Some(version) = self.commit_version(&txn_id)? {
            self.deltas.insert(txn_id, version);
        }

        self.commits.insert(txn_id);
        Ok(())
    }

    fn read_and_commit(&mut self, txn_id: I) -> CommitResult<K, V> {
        if self.check_committed(&txn_id)? {
            return Ok((self.canon(&txn_id), None));
        }

        self.commit(txn_id)?;
        Ok((self.canon(&txn_id), self.deltas.get(&txn_id).cloned()))
    }

    #[inline]
    fn contains_canon<Q>(&self, txn_id: &I, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        contains_canon(&self.canon, &self.deltas, txn_id, key)
    }

    #[inline]
    fn contains_committed<Q>(&self, txn_id: &I, key: &Q) -> Poll<Result<bool>>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        match self.finalized.as_ref().cmp(&Some(txn_id)) {
            Ordering::Greater => Poll::Ready(Err(Error::Outdated)),
            Ordering::Equal => Poll::Ready(Ok(self.contains_canon(txn_id, key))),
            Ordering::Less => {
                if self.commits.contains(txn_id) {
                    Poll::Ready(Ok(self.contains_canon(txn_id, key)))
                } else {
                    Poll::Pending
                }
            }
        }
    }

    #[inline]
    fn contains_pending<Q>(&self, txn_id: &I, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if let Some(delta) = self.pending.get(txn_id) {
            if let Some(key_state) = delta.get(key) {
                return key_state.is_some();
            }
        }

        self.contains_canon(txn_id, key)
    }

    #[inline]
    fn extend<Q, E>(&mut self, txn_id: I, other: E)
    where
        Q: Into<Key<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        let entries = other
            .into_iter()
            .map(|(key, value)| (key.into(), Some(Arc::new(RwLock::new(value)))));

        if let Some(version) = self.pending.get_mut(&txn_id) {
            version.extend(entries);
        } else {
            self.pending.insert(txn_id, entries.collect());
        }
    }

    #[inline]
    fn finalize(&mut self, txn_id: I) -> Option<&Canon<K, V>> {
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
            merge_owned(&mut self.canon, version);
        }

        self.finalized = Some(txn_id);

        Some(&self.canon)
    }

    #[inline]
    fn get_canon<Q>(&self, txn_id: &I, key: &Q) -> Option<Arc<V>>
    where
        Q: Hash + Eq + ?Sized,
        Key<K>: Borrow<Q>,
    {
        get_canon(&self.canon, &self.deltas, txn_id, key).cloned()
    }

    #[inline]
    fn get_committed<Q>(
        &self,
        txn_id: &I,
        key: &Q,
    ) -> Poll<Result<Option<TxnMapValueReadGuard<K, V>>>>
    where
        Q: Hash + Eq + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if self.finalized.as_ref() > Some(txn_id) {
            Poll::Ready(Err(Error::Outdated))
        } else if self.commits.contains(txn_id) {
            debug_assert!(!self.pending.contains_key(txn_id));
            let value = self.get_canon(txn_id, key);
            Poll::Ready(Ok(value.map(TxnMapValueReadGuard::committed)))
        } else {
            Poll::Pending
        }
    }

    #[inline]
    fn get_pending<Q>(&self, txn_id: &I, key: &Q) -> Option<PendingValue<V>>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if let Some(version) = self.pending.get(txn_id) {
            if let Some(delta) = version.get(key) {
                return if let Some(value) = delta {
                    // the permit means it's safe to call try_read_owned().expect()
                    let guard = value.clone().try_read_owned().expect("read version");
                    Some(PendingValue::Pending(guard))
                } else {
                    None
                };
            }
        }

        self.get_canon(txn_id, key)
            .map(|value| PendingValue::Committed(value))
    }

    #[inline]
    fn insert(&mut self, txn_id: I, key: Key<K>, value: V) -> Option<Arc<V>> {
        let value = Arc::new(RwLock::new(value));

        if let Some(deltas) = self.pending.get_mut(&txn_id) {
            match deltas.entry(key) {
                hash_map::Entry::Occupied(mut delta) => {
                    if let Some(prior) = delta.insert(Some(value)) {
                        let lock = Arc::try_unwrap(prior).expect("prior value");
                        let prior_value = Arc::new(lock.into_inner());
                        Some(prior_value)
                    } else {
                        get_canon(&self.canon, &self.deltas, &txn_id, delta.key()).cloned()
                    }
                }
                hash_map::Entry::Vacant(delta) => {
                    let prior = get_canon(&self.canon, &self.deltas, &txn_id, delta.key()).cloned();

                    delta.insert(Some(value));
                    prior
                }
            }
        } else {
            let prior = get_canon(&self.canon, &self.deltas, &txn_id, &key).cloned();

            let pending = iter::once((key, Some(value))).collect();
            self.pending.insert(txn_id, pending);

            prior
        }
    }

    #[inline]
    fn key<Q>(&self, txn_id: &I, key: &Q) -> Option<&Key<K>>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        if let Some((key, _)) = self.canon.get_key_value(key) {
            Some(key)
        } else if let Some(deltas) = self.pending.get(txn_id) {
            deltas.get_key_value(key).map(|(key, _delta)| key)
        } else {
            None
        }
    }

    #[inline]
    fn keys_committed(&self, txn_id: &I) -> HashSet<Key<K>> {
        let mut keys = self.canon.keys().cloned().collect();

        let deltas = self.deltas.range(..=txn_id).map(|(_, version)| version);

        for version in deltas {
            merge_keys(&mut keys, version);
        }

        keys
    }

    #[inline]
    fn keys_pending(&self, txn_id: I) -> HashSet<Key<K>> {
        let mut keys = self.keys_committed(&txn_id);

        if let Some(pending) = self.pending.get(&txn_id) {
            merge_keys(&mut keys, pending);
        }

        keys
    }

    #[inline]
    fn remove(&mut self, txn_id: I, key: Key<K>) -> Option<Arc<V>> {
        if let Some(pending) = self.pending.get_mut(&txn_id) {
            match pending.entry(key) {
                hash_map::Entry::Occupied(mut entry) => {
                    if let Some(lock) = entry.insert(None) {
                        let lock = Arc::try_unwrap(lock).expect("removed value");
                        Some(Arc::new(lock.into_inner()))
                    } else {
                        None
                    }
                }
                hash_map::Entry::Vacant(entry) => {
                    if let Some(prior) =
                        get_canon(&self.canon, &self.deltas, &txn_id, entry.key()).cloned()
                    {
                        entry.insert(None);
                        Some(prior)
                    } else {
                        None
                    }
                }
            }
        } else if let Some(prior) = get_canon(&self.canon, &self.deltas, &txn_id, &key).cloned() {
            let pending = iter::once((key, None)).collect();
            self.pending.insert(txn_id, pending);
            Some(prior)
        } else {
            None
        }
    }
}

#[inline]
fn contains_canon<I, K, V, Q>(
    canon: &Canon<K, V>,
    deltas: &BTreeMap<I, Delta<K, V>>,
    txn_id: &I,
    key: &Q,
) -> bool
where
    I: Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    Q: Eq + Hash + ?Sized,
    Key<K>: Borrow<Q>,
{
    let deltas = deltas.range(..=txn_id).rev().map(|(_, version)| version);

    for delta in deltas {
        if let Some(key_state) = delta.get(key) {
            return key_state.is_some();
        }
    }

    canon.contains_key(key)
}

impl<I, K, V> State<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: Clone + fmt::Debug,
{
    #[inline]
    fn get_mut(&mut self, txn_id: I, key: Key<K>) -> Option<OwnedRwLockWriteGuard<V>> {
        #[inline]
        fn new_value<V: Clone>(canon: &V) -> (Arc<RwLock<V>>, OwnedRwLockWriteGuard<V>) {
            let value = V::clone(canon);
            let value = Arc::new(RwLock::new(value));
            let guard = value.clone().try_write_owned().expect("write version");
            (value, guard)
        }

        if let Some(pending) = self.pending.get_mut(&txn_id) {
            match pending.entry(key) {
                hash_map::Entry::Occupied(delta) => {
                    let value = delta.get().as_ref()?;

                    value
                        .clone()
                        .try_write_owned()
                        .map(Some)
                        .expect("write version")
                }
                hash_map::Entry::Vacant(delta) => {
                    let canon = get_canon(&self.canon, &self.deltas, &txn_id, delta.key())?;
                    let (value, guard) = new_value(&**canon);

                    delta.insert(Some(value));

                    Some(guard)
                }
            }
        } else {
            let canon = get_canon(&self.canon, &self.deltas, &txn_id, &key)?;
            let (value, guard) = new_value(&**canon);

            let version = iter::once((key, Some(value))).collect();
            self.pending.insert(txn_id, version);

            Some(guard)
        }
    }

    #[inline]
    fn insert_new(&mut self, txn_id: I, key: Key<K>, value: V) -> OwnedRwLockWriteGuard<V> {
        let value = Arc::new(RwLock::new(value));
        let guard = value.clone().try_write_owned().expect("value");

        if let Some(version) = self.pending.get_mut(&txn_id) {
            assert!(version.insert(key, Some(value)).flatten().is_none());
        } else {
            let version = iter::once((key, Some(value))).collect();
            self.pending.insert(txn_id, version);
        }

        guard
    }
}

#[inline]
fn get_canon<'a, I, K, V, Q>(
    canon: &'a Canon<K, V>,
    deltas: &'a Deltas<I, K, V>,
    txn_id: &'a I,
    key: &'a Q,
) -> Option<&'a Arc<V>>
where
    I: Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    Q: Eq + Hash + ?Sized,
    Key<K>: Borrow<Q>,
{
    let committed = deltas.range(..=txn_id).rev().map(|(_, version)| version);

    for version in committed {
        if let Some(delta) = version.get(key) {
            return delta.as_ref();
        }
    }

    canon.get(key)
}

#[inline]
fn merge_keys<K: Hash + Ord, V>(keys: &mut HashSet<Key<K>>, deltas: &Delta<K, V>) {
    for (key, delta) in deltas {
        if delta.is_some() {
            keys.insert(key.clone());
        } else {
            keys.remove(key);
        }
    }
}

/// An occupied entry in a [`TxnMapLock`]
pub struct EntryOccupied<K, V> {
    key: Key<K>,
    value: TxnMapValueWriteGuard<K, V>,
}

impl<K, V> EntryOccupied<K, V> {
    /// Borrow this entry's value.
    pub fn get(&self) -> &V {
        self.value.deref()
    }

    /// Borrow this entry's value mutably.
    pub fn get_mut(&mut self) -> &mut V {
        self.value.deref_mut()
    }

    /// Borrow this entry's key.
    pub fn key(&self) -> &K {
        &self.key
    }
}

/// A vacant entry in a [`TxnMapLock`]
pub struct EntryVacant<I, K, V> {
    txn_id: I,
    key: Key<K>,
    map_state: Arc<RwLockInner<State<I, K, V>>>,
    permit: PermitWrite<Range<K>>,
}

impl<I, K, V> EntryVacant<I, K, V>
where
    I: Hash + Ord + Copy + fmt::Debug,
    K: Hash + Ord + Clone + fmt::Debug,
    V: Clone + fmt::Debug,
{
    /// Inspect the committed value hidden by this transaction's removal, if any.
    /// Reject a newer committed change to this key with [`Error::Conflict`], so
    /// a native owner cannot be recreated from obsolete membership.
    /// The entry retains its existing write permit; the returned value is shared.
    pub fn get_committed(&self) -> Result<Option<Arc<V>>> {
        let state = self
            .map_state
            .read()
            .expect("transaction bookkeeping poisoned");

        state.check_pending(&self.txn_id)?;
        if state
            .deltas
            .range((
                std::ops::Bound::Excluded(self.txn_id),
                std::ops::Bound::Unbounded,
            ))
            .any(|(_, version)| version.contains_key(&self.key))
        {
            return Err(Error::Conflict);
        }

        Ok(state.get_canon(&self.txn_id, &self.key))
    }

    /// Insert a new value at this [`Entry`].
    pub fn insert(self, value: V) -> Result<TxnMapValueWriteGuard<K, V>> {
        let mut map_state = self
            .map_state
            .write()
            .expect("transaction bookkeeping poisoned");

        map_state.check_pending(&self.txn_id)?;
        let value = map_state.insert_new(self.txn_id, self.key, value);
        Ok(TxnMapValueWriteGuard::new(self.permit, value))
    }

    /// Borrow this entry's key.
    pub fn key(&self) -> &K {
        &self.key
    }
}

/// An entry in a [`TxnMapLock`]
pub enum Entry<I, K, V> {
    Occupied(EntryOccupied<K, V>),
    Vacant(EntryVacant<I, K, V>),
}

impl<I, K, V> Entry<I, K, V> {
    /// Borrow the key of either an occupied or vacant entry.
    pub fn key(&self) -> &K {
        match self {
            Self::Occupied(entry) => &entry.key,
            Self::Vacant(entry) => &entry.key,
        }
    }
}

/// A futures-aware read-write lock on a [`HashMap`] which supports transactional versioning
///
/// The `get_mut` and `try_get_mut` methods require the value type `V` to implement [`Clone`]
/// in order to support multiple transactions with different versions.
pub struct TxnMapLock<I, K, V> {
    state: Arc<RwLockInner<State<I, K, V>>>,
    semaphore: Semaphore<I, Collator<K>, Range<K>>,
}

impl<I, K, V> Clone for TxnMapLock<I, K, V> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<I, K, V> TxnMapLock<I, K, V> {
    #[inline]
    fn state(&self) -> impl Deref<Target = State<I, K, V>> + '_ {
        self.state.read().expect("transaction bookkeeping poisoned")
    }

    #[inline]
    fn state_mut(&self) -> impl DerefMut<Target = State<I, K, V>> + '_ {
        self.state
            .write()
            .expect("transaction bookkeeping poisoned")
    }
}

impl<I: Ord + Hash + fmt::Debug, K, V> TxnMapLock<I, K, V> {
    /// Construct a new [`TxnMapLock`].
    pub fn new(txn_id: I) -> Self {
        let collator = Collator::<K>::default();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, Pending::new()))),
            semaphore: Semaphore::new(collator),
        }
    }
}

impl<I, K, V> TxnMapLock<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Eq + Hash + Ord + fmt::Debug + Send + Sync,
    V: fmt::Debug,
{
    /// Construct a new [`TxnMapLock`] with the given `contents`.
    pub fn with_contents<KV: IntoIterator<Item = (K, V)>>(txn_id: I, contents: KV) -> Self {
        let version = contents
            .into_iter()
            .map(|(key, value)| (Key::new(key), Some(Arc::new(RwLock::new(value)))))
            .collect();

        let collator = Collator::<K>::default();

        Self {
            state: Arc::new(RwLockInner::new(State::new(txn_id, version))),
            semaphore: Semaphore::with_reservation(txn_id, collator, Range::All),
        }
    }

    /// Construct a transactional map whose initial contents are already committed.
    /// This does not reserve, commit, or finalize a synthetic transaction version.
    pub fn from_committed<KV: IntoIterator<Item = (K, V)>>(contents: KV) -> Self {
        let canon = contents
            .into_iter()
            .map(|(key, value)| (Key::new(key), Arc::new(value)))
            .collect();

        Self {
            state: Arc::new(RwLockInner::new(State::from_committed(canon))),
            semaphore: Semaphore::new(Collator::<K>::default()),
        }
    }

    /// Remove and return all entries from this [`TxnMapLock`] at `txn_id`.
    pub async fn clear(&self, txn_id: I) -> Result<Canon<K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("clear {self:?} at {txn_id:?}");

        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.clear(txn_id))
    }

    /// Remove and return all entries from this [`TxnMapLock`] at `txn_id`.
    pub fn try_clear(&self, txn_id: I) -> Result<Canon<K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("try_clear {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        Ok(state.clear(txn_id))
    }

    /// Return `true` if this [`TxnMapLock`] has an entry at `key` at `txn_id`.
    pub async fn contains_key<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("contains_key? {self:?} at {txn_id:?}");

        let range: Range<K> = {
            let state = self.state();
            if let Poll::Ready(result) = state.contains_committed(&txn_id, key) {
                return result;
            } else {
                Key::<K>::from((key, state.key(&txn_id, key))).into()
            }
        };

        let _permit = self.semaphore.read(txn_id, range).await?;
        Ok(self.state().contains_pending(&txn_id, key))
    }

    /// Synchronously check whether the given `key` is present in this [`TxnMapLock`]s, if possible.
    pub fn try_contains_key<Q>(&self, txn_id: I, key: &Q) -> Result<bool>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_contains_key {self:?} at {txn_id:?}");

        let state = self.state();
        if let Poll::Ready(result) = state.contains_committed(&txn_id, key) {
            return result;
        }

        let range = Key::<K>::from((key, state.key(&txn_id, key))).into();
        let _permit = self.semaphore.try_read(txn_id, range)?;
        Ok(state.contains_pending(&txn_id, key))
    }

    /// Insert the entries from `other` [`TxnMapLock`] at `txn_id`.
    pub async fn extend<Q, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        Q: Into<Key<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        #[cfg(feature = "logging")]
        log::trace!("extend {self:?} at {txn_id:?}");

        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.write(txn_id, Range::All).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        state.extend(txn_id, other);
        Ok(())
    }

    /// Insert the entries from `other` [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_extend<Q, E>(&self, txn_id: I, other: E) -> Result<()>
    where
        Q: Into<Key<K>>,
        E: IntoIterator<Item = (Q, V)>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_extend {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let _permit = self.semaphore.try_write(txn_id, Range::All)?;

        state.extend(txn_id, other);
        Ok(())
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get<Q>(&self, txn_id: I, key: &Q) -> Result<Option<TxnMapValueReadGuard<K, V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("get from {self:?} at {txn_id:?}");

        let range: Range<K> = {
            let state = self.state();

            if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
                return result;
            }

            Key::<K>::from((key, state.key(&txn_id, key))).into()
        };

        let permit = self.semaphore.read(txn_id, range).await?;

        let state = self.state();

        if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
            return result;
        }

        let value = state.get_pending(&txn_id, key).map(|value| match value {
            PendingValue::Committed(value) => TxnMapValueReadGuard::committed(value),
            PendingValue::Pending(value) => TxnMapValueReadGuard::pending_write(permit, value),
        });

        Ok(value)
    }

    /// Read a value from this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_get<Q>(&self, txn_id: I, key: &Q) -> Result<Option<TxnMapValueReadGuard<K, V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_get from {self:?} at {txn_id:?}");

        let state = self.state();

        if let Poll::Ready(result) = state.get_committed(&txn_id, key) {
            #[cfg(feature = "log")]
            log::trace!(
                "{self:?} is already committed at {txn_id:?}, no need for a read permit..."
            );

            return result;
        } else {
            #[cfg(feature = "log")]
            log::trace!("{self:?} is not yet committed at {txn_id:?}, getting a read permit...");
        }

        let range = Key::<K>::from((key, state.key(&txn_id, key))).into();
        let permit = self.semaphore.try_read(txn_id, range)?;
        let value = state.get_pending(&txn_id, key).map(|value| match value {
            PendingValue::Committed(value) => TxnMapValueReadGuard::committed(value),
            PendingValue::Pending(value) => TxnMapValueReadGuard::pending_write(permit, value),
        });

        Ok(value)
    }

    /// Construct an iterator over the entries in this [`TxnMapLock`] at `txn_id`.
    pub async fn iter(&self, txn_id: I) -> Result<Iter<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("iter over {self:?} at {txn_id:?}");

        let permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();

        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, None, keys))
        } else {
            let keys = self.state().keys_pending(txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, Some(permit), keys))
        }
    }

    /// Construct an iterator over the entries in this [`TxnMapLock`] at `txn_id` synchronously.
    pub fn try_iter(&self, txn_id: I) -> Result<Iter<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("try_iter over {self:?} at {txn_id:?}");

        let state = self.state();

        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, None, keys))
        } else {
            let permit = self.semaphore.try_read(txn_id, Range::All)?;
            let keys = state.keys_pending(txn_id);
            Ok(Iter::new(self.state.clone(), txn_id, Some(permit), keys))
        }
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` and return the prior value, if any.
    pub async fn insert<Q: Into<Key<K>>>(
        &self,
        txn_id: I,
        key: Q,
        value: V,
    ) -> Result<Option<Arc<V>>> {
        #[cfg(feature = "logging")]
        log::trace!("insert into {self:?} at {txn_id:?}");

        let key: Key<K> = key.into();
        self.state().check_pending(&txn_id)?;

        let _permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.insert(txn_id, key, value))
    }

    /// Insert a new entry into this [`TxnMapLock`] at `txn_id` synchronously, if possible.
    pub fn try_insert<Q: Into<Key<K>>>(
        &self,
        txn_id: I,
        key: Q,
        value: V,
    ) -> Result<Option<Arc<V>>> {
        #[cfg(feature = "logging")]
        log::trace!("try_insert into {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let key = key.into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        Ok(state.insert(txn_id, key, value))
    }

    /// Return `true` if this [`TxnMapLock`] is empty at the given `txn_id`.
    pub async fn is_empty(&self, txn_id: I) -> Result<bool> {
        #[cfg(feature = "logging")]
        log::trace!("is_empty? {self:?} at {txn_id:?}");

        self.len(txn_id).map_ok(|len| len == 0).await
    }

    /// Get the size of this [`TxnMapLock`] at the given `txn_id`.
    pub async fn len(&self, txn_id: I) -> Result<usize> {
        #[cfg(feature = "logging")]
        log::trace!("len of {self:?} at {txn_id:?}");

        let _permit = self.semaphore.read(txn_id, Range::All).await?;

        let state = self.state();

        if state.check_committed(&txn_id)? {
            let keys = state.keys_committed(&txn_id);
            Ok(keys.len())
        } else {
            let keys = state.keys_pending(txn_id);
            Ok(keys.len())
        }
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub async fn remove<Q>(&self, txn_id: I, key: &Q) -> Result<Option<Arc<V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("remove from {self:?} at {txn_id:?}");

        let key: Key<K> = {
            let state = self.state();
            state.check_pending(&txn_id)?;
            (key, state.key(&txn_id, key)).into()
        };

        let _permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        Ok(state.remove(txn_id, key))
    }

    /// Remove and return the value at `key` from this [`TxnMapLock`] at `txn_id`, if present.
    pub fn try_remove<Q>(&self, txn_id: I, key: &Q) -> Result<Option<Arc<V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_remove from {self:?} at {txn_id:?}");

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;

        let key: Key<K> = (key, state.key(&txn_id, key)).into();
        let _permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        Ok(state.remove(txn_id, key))
    }
}

impl<I, K, V> TxnMapLock<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord + fmt::Debug + Send + Sync,
    V: Clone + fmt::Debug,
{
    /// Borrow an [`Entry`] mutably for writing at `txn_id`.
    pub async fn entry<Q: Into<Key<K>>>(&self, txn_id: I, key: Q) -> Result<Entry<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("get entry in {self:?} at {txn_id:?}");

        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key: Key<K> = key.into();
        let range = key.clone().into();
        let permit = self.semaphore.write(txn_id, range).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;
        if let Some(value) = state.get_mut(txn_id, key.clone()) {
            Ok(Entry::Occupied(EntryOccupied {
                key,
                value: TxnMapValueWriteGuard::new(permit, value),
            }))
        } else {
            Ok(Entry::Vacant(EntryVacant {
                permit,
                key,
                txn_id,
                map_state: self.state.clone(),
            }))
        }
    }

    /// Read a mutable value from this [`TxnMapLock`] at `txn_id`.
    pub async fn get_mut<Q: Into<Key<K>>>(
        &self,
        txn_id: I,
        key: Q,
    ) -> Result<Option<TxnMapValueWriteGuard<K, V>>> {
        #[cfg(feature = "logging")]
        log::trace!("get_mut {self:?} at {txn_id:?}");

        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let key = key.into();
        let permit = self.semaphore.write(txn_id, key.clone().into()).await?;

        let mut state = self.state_mut();
        state.check_pending(&txn_id)?;
        if let Some(value) = state.get_mut(txn_id, key) {
            Ok(Some(TxnMapValueWriteGuard::new(permit, value)))
        } else {
            Ok(None)
        }
    }

    /// Read a mutable value from this [`TxnMapLock`] at `txn_id`.
    pub fn try_get_mut<Q>(&self, txn_id: I, key: &Q) -> Result<Option<TxnMapValueWriteGuard<K, V>>>
    where
        Q: Eq + Hash + ToOwned<Owned = K> + ?Sized,
        Key<K>: Borrow<Q>,
    {
        #[cfg(feature = "logging")]
        log::trace!("try_get_mut {self:?} at {txn_id:?}");

        let mut state = self.state_mut();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let maybe_key = state.key(&txn_id, key);
        let key = Key::<K>::from((key, maybe_key));
        let permit = self.semaphore.try_write(txn_id, key.clone().into())?;

        if let Some(value) = state.get_mut(txn_id, key) {
            Ok(Some(TxnMapValueWriteGuard::new(permit, value)))
        } else {
            Ok(None)
        }
    }

    /// Construct a mutable iterator over the entries in this [`TxnMapLock`] at `txn_id`.
    pub async fn iter_mut(&self, txn_id: I) -> Result<IterMut<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("iter_mut {self:?} at {txn_id:?}");

        // before acquiring a permit, check if this version has already been committed
        self.state().check_pending(&txn_id)?;

        let permit = self.semaphore.write(txn_id, Range::All).await?;
        let state = self.state();
        state.check_pending(&txn_id)?;
        let keys = state.keys_pending(txn_id);
        Ok(IterMut::new(self.state.clone(), txn_id, permit, keys))
    }

    /// Construct a mutable iterator over the entries in this [`TxnMapLock`] at `txn_id`,
    /// synchronously if possible.
    pub fn try_iter_mut(&self, txn_id: I) -> Result<IterMut<I, K, V>> {
        #[cfg(feature = "logging")]
        log::trace!("try_iter_mut {self:?} at {txn_id:?}");

        let state = self.state();

        // before acquiring a permit, check if this version has already been committed
        state.check_pending(&txn_id)?;

        let permit = self.semaphore.try_write(txn_id, Range::All)?;
        let keys = state.keys_pending(txn_id);
        Ok(IterMut::new(self.state.clone(), txn_id, permit, keys))
    }
}

impl<I, K, V> TxnMapLock<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Eq + Hash + Ord + fmt::Debug + Send + Sync,
    V: fmt::Debug,
{
    /// Inspect whether this transaction has committed, without read admission.
    /// IDs below the finalized cutoff return [`Error::Outdated`]; the cutoff
    /// itself is committed. Callers coordinate external publication separately.
    pub fn is_committed(&self, txn_id: I) -> Result<bool> {
        self.state().check_committed(&txn_id)
    }

    /// Check decision eligibility without changing state or acquiring admission.
    /// This checks the finalized cutoff and outstanding pending-value borrows;
    /// it does not reserve a decision or replace the caller's quiescence obligation.
    pub fn check_commit(&self, txn_id: I) -> Result<()> {
        self.state().check_commit(&txn_id)
    }

    /// Commit the state of this [`TxnMapLock`] at `txn_id`.
    /// Returns an error for an invalid decision or unavailable state.
    pub fn commit(&self, txn_id: I) -> Result<()> {
        #[cfg(feature = "logging")]
        log::trace!("commit {self:?} at {txn_id:?}");

        let mut state = self.state_mut();

        if state.finalized == Some(txn_id) {
            return Ok(());
        }

        if state.finalized.as_ref() > Some(&txn_id) {
            return Err(Error::Outdated);
        }

        state.commit(txn_id)?;

        self.semaphore.finalize(&txn_id, false)?;
        Ok(())
    }

    /// Read and commit this [`TxnMapLock`] in a single operation.
    /// Also returns the changes newly committed by this call, if any.
    /// Repeated decisions return the historical snapshot without repeating changes.
    ///
    /// Returns an error for an invalid decision or unavailable state.
    pub async fn read_and_commit(&self, txn_id: I) -> CommitResult<K, V> {
        #[cfg(feature = "logging")]
        log::trace!("read and commit {self:?} at {txn_id:?}");

        // Reject a known invalid decision before registering a reservation.
        {
            let state = self.state();
            if state.check_committed(&txn_id)? {
                return Ok((state.canon(&txn_id), None));
            }
        }

        let _permit = self.semaphore.read(txn_id, Range::All).await?;

        let result = self.state_mut().read_and_commit(txn_id)?;
        self.semaphore.finalize(&txn_id, false)?;
        Ok(result)
    }

    /// Read and commit synchronously, returning [`Error::WouldBlock`] if causal
    /// read admission is unavailable. A rejected call does not commit or release
    /// reservations. Successful and repeated results match [`Self::read_and_commit`].
    pub fn try_read_and_commit(&self, txn_id: I) -> CommitResult<K, V> {
        let mut state = self.state_mut();
        if state.check_committed(&txn_id)? {
            return Ok((state.canon(&txn_id), None));
        }

        state.check_commit(&txn_id)?;

        let _permit = self.semaphore.try_read(txn_id, Range::All)?;
        let result = state.read_and_commit(txn_id)?;
        drop(state);
        self.semaphore.finalize(&txn_id, false)?;
        Ok(result)
    }

    /// Roll back the state of this [`TxnMapLock`] at `txn_id`.
    ///
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

    /// Read and roll back this [`TxnMapLock`] in a single operation.
    ///
    /// Returns discarded membership (the version that would have been committed),
    /// sharing values without cloning payloads. The restored state remains available
    /// through ordinary transactional reads.
    /// Returns an error for an invalid decision or unavailable state.
    pub async fn read_and_rollback(&self, txn_id: I) -> Result<Canon<K, V>> {
        // Reject a known invalid decision before registering a reservation.
        {
            let state = self.state();
            if state.finalized == Some(txn_id) {
                return Ok(state.canon(&txn_id));
            }

            if state.commits.contains(&txn_id) {
                return Err(Error::Committed);
            }

            if state.finalized > Some(txn_id) {
                return Err(Error::Outdated);
            }
        }

        let _permit = self.semaphore.read(txn_id, Range::All).await?;

        let discarded = {
            let mut state = self.state_mut();
            if state.finalized == Some(txn_id) {
                return Ok(state.canon(&txn_id));
            }

            if state.commits.contains(&txn_id) {
                return Err(Error::Committed);
            }

            if state.finalized > Some(txn_id) {
                return Err(Error::Outdated);
            }

            let deltas = state.commit_version(&txn_id)?;
            let mut discarded = state.canon(&txn_id);
            if let Some(deltas) = deltas {
                merge_owned(&mut discarded, deltas);
            }

            discarded
        };

        self.semaphore.finalize(&txn_id, false)?;

        Ok(discarded)
    }

    /// Finalize the state of this [`TxnMapLock`] at `txn_id`.
    /// This will finalize commits and prevent further reads of all versions earlier than `txn_id`.
    pub fn finalize(&self, txn_id: I) -> Result<()> {
        let mut state = self.state_mut();
        self.semaphore.finalize(&txn_id, true)?;
        state.finalize(txn_id);
        Ok(())
    }

    /// Read and finalize the state of this [`TxnMapLock`] at `txn_id`, if not already finalized.
    /// This will finalize commits and prevent further reads of all versions earlier than `txn_id`.
    pub fn read_and_finalize(&self, txn_id: I) -> Result<Option<Canon<K, V>>> {
        let mut state = self.state_mut();
        self.semaphore.finalize(&txn_id, true)?;
        Ok(state.finalize(txn_id).cloned())
    }

    /// Inspect this key's finalized baseline, latest committed value, and cutoff.
    ///
    /// Pending values are excluded, including when they are borrowed. This is
    /// metadata inspection, not transactional read admission: the caller retains
    /// its transactional guard and separately protects native publication. The
    /// callback runs under bookkeeping protection and must not block or reenter.
    pub fn inspect_committed<Q, R>(
        &self,
        txn_id: I,
        key: &Q,
        inspect: impl FnOnce(Option<&V>, Option<&V>, Option<I>) -> R,
    ) -> Result<R>
    where
        Q: Eq + Hash + ?Sized,
        Key<K>: Borrow<Q>,
    {
        let state = self.state();
        if state.finalized > Some(txn_id) {
            return Err(Error::Outdated);
        }

        let baseline = state.canon.get(key).map(|value| value.as_ref());
        let latest = state
            .deltas
            .values()
            .rev()
            .find_map(|version| version.get(key))
            .map_or(baseline, |value| value.as_deref());

        Ok(inspect(baseline, latest, state.finalized))
    }

    /// Inspect retained canonical, committed, and pending entries synchronously.
    ///
    /// `None` identifies the finalized baseline; other IDs identify retained
    /// versions, including removals. Pending values in use return `WouldBlock`.
    /// The callback must not reenter this lock. Callers coordinating native
    /// storage must protect publication separately; no guard escapes this call.
    pub fn inspect_versions(
        &self,
        txn_id: I,
        mut inspect: impl FnMut(Option<I>, &K, Option<&V>),
    ) -> Result<()> {
        let state = self.state();
        if state.finalized > Some(txn_id) {
            return Err(Error::Outdated);
        }

        for (key, value) in &state.canon {
            inspect(None, key, Some(value));
        }

        for (id, version) in &state.deltas {
            for (key, value) in version {
                inspect(Some(*id), key, value.as_deref());
            }
        }

        for (id, version) in &state.pending {
            for (key, value) in version {
                if let Some(value) = value {
                    let value = value.try_read().map_err(|_| Error::WouldBlock)?;
                    inspect(Some(*id), key, Some(&value));
                } else {
                    inspect(Some(*id), key, None);
                }
            }
        }

        Ok(())
    }
}

impl<I, K, V> fmt::Debug for TxnMapLock<I, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a transactional lock on a map of keys to values")
    }
}

/// A guard on a value in an [`Iter`]. Pending native access retains its permit
/// independently of the iterator, and drops before releasing that permit.
#[derive(Debug)]
pub struct TxnMapLockIterGuard<K, V> {
    value: PendingValue<V>,
    _permit: Option<Arc<PermitRead<Range<K>>>>,
}

impl<K, V> Deref for TxnMapLockIterGuard<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        match &self.value {
            PendingValue::Committed(value) => value.deref(),
            PendingValue::Pending(value) => value.deref(),
        }
    }
}

impl<K, V: PartialEq> PartialEq<V> for TxnMapLockIterGuard<K, V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

impl<K, V: PartialOrd> PartialOrd<V> for TxnMapLockIterGuard<K, V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

/// An iterator over the entries in a [`TxnMapLock`] as of a specific transaction
pub struct Iter<I, K, V> {
    lock_state: Arc<RwLockInner<State<I, K, V>>>,
    txn_id: I,
    keys: <HashSet<Key<K>> as IntoIterator>::IntoIter,
    permit: Option<Arc<PermitRead<Range<K>>>>,
}

impl<I, K, V> Iter<I, K, V> {
    fn new(
        lock_state: Arc<RwLockInner<State<I, K, V>>>,
        txn_id: I,
        permit: Option<PermitRead<Range<K>>>,
        keys: HashSet<Key<K>>,
    ) -> Self {
        Self {
            lock_state,
            txn_id,
            permit: permit.map(Arc::new),
            keys: keys.into_iter(),
        }
    }
}

impl<I, K, V> Iterator for Iter<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: fmt::Debug,
{
    type Item = (Key<K>, TxnMapLockIterGuard<K, V>);

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.lock_state.read().expect("lock state");

        loop {
            let key = self.keys.next()?;
            let value = get_key(&state, &self.txn_id, &key, self.permit.is_none());
            if let Some(value) = value {
                let permit = matches!(&value, PendingValue::Pending(_))
                    .then(|| self.permit.clone())
                    .flatten();

                return Some((
                    key,
                    TxnMapLockIterGuard {
                        value,
                        _permit: permit,
                    },
                ));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

#[inline]
fn get_key<I, K, V, Q>(
    state: &State<I, K, V>,
    txn_id: &I,
    key: &Q,
    committed: bool,
) -> Option<PendingValue<V>>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: fmt::Debug,
    Q: Eq + Hash + ?Sized,
    Key<K>: Borrow<Q>,
{
    if committed {
        state.get_canon(txn_id, key).map(PendingValue::Committed)
    } else {
        state.get_pending(txn_id, key)
    }
}

/// A guard on a mutable value in an [`IterMut`]. It retains write authority
/// until native access is released, even after its iterator is dropped.
#[derive(Debug)]
pub struct TxnMapLockIterMutGuard<K, V> {
    guard: OwnedRwLockWriteGuard<V>,
    _permit: Arc<PermitWrite<Range<K>>>,
}

impl<K, V> Deref for TxnMapLockIterMutGuard<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<K, V> DerefMut for TxnMapLockIterMutGuard<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<K, V: PartialEq> PartialEq<V> for TxnMapLockIterMutGuard<K, V> {
    fn eq(&self, other: &V) -> bool {
        self.deref().eq(other)
    }
}

impl<K, V: PartialOrd> PartialOrd<V> for TxnMapLockIterMutGuard<K, V> {
    fn partial_cmp(&self, other: &V) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}

/// An iterator over the keys and mutable values in a [`TxnMapLock`]
pub struct IterMut<I, K, V> {
    lock_state: Arc<RwLockInner<State<I, K, V>>>,
    txn_id: I,

    keys: <HashSet<Key<K>> as IntoIterator>::IntoIter,
    _permit: Arc<PermitWrite<Range<K>>>,
}

impl<I, K, V> IterMut<I, K, V> {
    fn new(
        lock_state: Arc<RwLockInner<State<I, K, V>>>,
        txn_id: I,
        permit: PermitWrite<Range<K>>,
        keys: HashSet<Key<K>>,
    ) -> Self {
        Self {
            lock_state,
            txn_id,
            _permit: Arc::new(permit),
            keys: keys.into_iter(),
        }
    }
}

impl<I, K, V> Iterator for IterMut<I, K, V>
where
    I: Copy + Hash + Ord + fmt::Debug,
    K: Hash + Ord,
    V: Clone + fmt::Debug,
{
    type Item = (Arc<K>, TxnMapLockIterMutGuard<K, V>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut state = self.lock_state.write().expect("lock state");

        loop {
            let key = self.keys.next()?;
            if let Some(guard) = state.get_mut(self.txn_id, key.clone()) {
                return Some((
                    key.into(),
                    TxnMapLockIterMutGuard {
                        guard,
                        _permit: self._permit.clone(),
                    },
                ));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

#[inline]
fn merge_owned<K: Eq + Hash, V>(version: &mut Canon<K, V>, deltas: Delta<K, V>) {
    for (key, delta) in deltas {
        match delta {
            Some(value) => version.insert(key, value),
            None => version.remove(&key),
        };
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::Duration;

    use super::TxnMapLock;
    use crate::Error;

    #[tokio::test]
    async fn transactional_iteration_observes_ordered_versions() {
        let map = TxnMapLock::<u64, String, u64>::from_committed([("base".into(), 0)]);
        map.insert(2, "pending".to_string(), 2).await.unwrap();
        let before = map
            .iter(1)
            .await
            .unwrap()
            .map(|(key, _)| (*key).clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(before, BTreeSet::from(["base".to_string()]));

        map.rollback(&2).unwrap();
        map.insert(3, "committed".to_string(), 3).await.unwrap();
        map.commit(3).unwrap();
        let before_finalize = map
            .iter(4)
            .await
            .unwrap()
            .map(|(key, _)| (*key).clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(
            before_finalize,
            BTreeSet::from(["base".to_string(), "committed".to_string()])
        );

        map.finalize(3).unwrap();
        let after_finalize = map
            .iter(4)
            .await
            .unwrap()
            .map(|(key, _)| (*key).clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(after_finalize, before_finalize);
    }

    #[tokio::test]
    async fn transactional_iteration_holds_its_read_range() {
        let map = TxnMapLock::<u64, String, u64>::from_committed([("base".into(), 0)]);
        {
            let mut snapshot = map.iter(2).await.unwrap();
            assert!(snapshot.next().is_some());
            let conflict = tokio::time::timeout(
                Duration::from_millis(100),
                map.insert(1, "too-late".to_string(), 1),
            )
            .await
            .expect("an older write must not wait behind a newer read")
            .expect_err("an older write must conflict with the held snapshot");

            assert_eq!(conflict, Error::Conflict);
        }
    }
}
