/// Transactional queue structures.
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::Hash;

use super::Error;

pub mod message;
pub mod task;

struct State<I, S> {
    pending: BTreeMap<I, S>,
    commits: BTreeSet<I>,
    finalized: Option<I>,
}

impl<I, S> State<I, S> {
    fn new() -> Self {
        Self {
            pending: BTreeMap::new(),
            commits: BTreeSet::new(),
            finalized: None,
        }
    }
}

impl<I, S> State<I, S>
where
    I: Eq + Hash + Ord,
{
    fn check_pending(&mut self, txn_id: I) -> Result<Entry<'_, I, S>, Error> {
        if Some(&txn_id) <= self.finalized.as_ref() {
            Err(Error::Outdated)
        } else if self.commits.contains(&txn_id) {
            Err(Error::Committed)
        } else {
            Ok(self.pending.entry(txn_id))
        }
    }
}

impl<I, S> State<I, S>
where
    I: Eq + Hash + Ord + fmt::Debug,
{
    fn commit(&mut self, txn_id: I) -> Result<Option<S>, Error> {
        if self.finalized.as_ref() >= Some(&txn_id) {
            return Err(Error::Outdated);
        }

        let queue = self.pending.remove(&txn_id);
        self.commits.insert(txn_id);
        Ok(queue)
    }

    fn rollback(&mut self, txn_id: &I) -> Result<Option<S>, Error> {
        if self.finalized.as_ref() >= Some(txn_id) {
            return Err(Error::Outdated);
        }

        if self.commits.contains(txn_id) {
            return Err(Error::Committed);
        }

        Ok(self.pending.remove(txn_id))
    }

    fn finalize(&mut self, txn_id: I) {
        while self.commits.first().is_some_and(|id| id <= &txn_id) {
            self.commits.pop_first();
        }

        while let Some(entry) = self.pending.first_entry() {
            if entry.key() > &txn_id {
                break;
            }

            entry.remove_entry();
        }

        if self.finalized.as_ref() > Some(&txn_id) {
            // no-op
        } else {
            self.finalized = Some(txn_id);
        }
    }
}
