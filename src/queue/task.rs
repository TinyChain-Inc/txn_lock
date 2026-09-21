//! A bounded queue of caller-driven transactional tasks.
//!
//! The caller owns and polls each task. Permits never spawn work or hold a mutex
//! across execution. Dropping an unfinished task fails its transaction.
//!
//! ```
//! use txn_lock::queue::task::TaskQueue;
//!
//! let queue = TaskQueue::<u64, &str>::new(16);
//! let mut task = queue.start(1).unwrap();
//! task.record("mutation").unwrap();
//! // Execute the operation represented by the record here.
//! task.complete().unwrap();
//! let mut operation = queue.operation().unwrap();
//! assert_eq!(operation.commit(&1).unwrap(), Some(vec!["mutation"]));
//! operation.complete();
//! ```

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::Error;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Active,
    Running,
    Failed,
    Committed,
    RolledBack,
}

struct Entry<O> {
    status: Status,
    outputs: Vec<O>,
}

#[derive(PartialEq, Eq)]
enum Owner {
    Ready,
    Busy,
    Interrupted,
}

struct State<I, O> {
    owner: Owner,
    finalized: Option<I>,
    entries: BTreeMap<I, Entry<O>>,
}

impl<I: Ord, O> State<I, O> {
    fn ready(&self) -> Result<(), Error> {
        match self.owner {
            Owner::Ready => Ok(()),
            Owner::Busy => Err(Error::Busy),
            Owner::Interrupted => Err(Error::Interrupted),
        }
    }

    fn check(&self, id: &I) -> Result<(), Error> {
        if self.owner == Owner::Interrupted {
            Err(Error::Interrupted)
        } else if self.finalized.as_ref().is_some_and(|cutoff| id <= cutoff) {
            Err(Error::Outdated)
        } else {
            Ok(())
        }
    }

    fn admit(&mut self, id: I) -> Result<&mut Entry<O>, Error> {
        self.check(&id)?;
        Ok(self.entries.entry(id).or_insert_with(|| Entry {
            status: Status::Active,
            outputs: vec![],
        }))
    }

    fn entry(&mut self, id: &I) -> Result<&mut Entry<O>, Error> {
        self.check(id)?;
        self.entries.get_mut(id).ok_or(Error::Conflict)
    }
}

/// Pending outputs and live decisions for one transactional owner.
/// Capacity bounds outputs per transaction; the caller separately bounds live transactions.
/// Clones share one owner. An interrupted operation makes every clone unusable.
pub struct TaskQueue<I, O> {
    capacity: usize,
    state: Arc<Mutex<State<I, O>>>,
}

impl<I, O> Clone for TaskQueue<I, O> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            state: self.state.clone(),
        }
    }
}

impl<I: Ord, O> TaskQueue<I, O> {
    /// Create an unpublished queue with a positive per-transaction capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "task queue capacity must be positive");
        Self {
            capacity,
            state: Arc::new(Mutex::new(State {
                owner: Owner::Ready,
                finalized: None,
                entries: BTreeMap::new(),
            })),
        }
    }

    fn state(&self) -> MutexGuard<'_, State<I, O>> {
        self.state.lock().expect("task queue state")
    }

    /// Require an unused queue before delegating it to a new owner.
    pub fn validate_fresh(&self) -> Result<(), Error> {
        let state = self.state();
        if state.owner == Owner::Ready && state.finalized.is_none() && state.entries.is_empty() {
            Ok(())
        } else {
            Err(Error::Conflict)
        }
    }

    /// Register an identity without retaining the caller's transaction capability.
    pub fn register(&self, id: I) -> Result<(), Error> {
        self.state().admit(id).map(|_| ())
    }

    /// Check an observation without registration, rejecting failed or rolled-back transactions.
    pub fn readable(&self, id: I) -> Result<(), Error> {
        let state = self.state();
        state.check(&id)?;
        match state.entries.get(&id).map(|entry| entry.status) {
            Some(Status::Failed) => Err(Error::Failed),
            Some(Status::RolledBack) => Err(Error::RolledBack),
            _ => Ok(()),
        }
    }

    /// Reserve one output before preparation. Recording releases preparation exclusion;
    /// execution then overlaps other transactions. Requests within one transaction
    /// remain ordered so execution and replay cannot disagree.
    pub fn start(&self, id: I) -> Result<Task<'_, I, O>, Error>
    where
        I: Clone,
    {
        {
            let mut state = self.state();
            state.ready()?;
            let entry = state.admit(id.clone())?;
            match entry.status {
                Status::Active => {}
                Status::Running => return Err(Error::Busy),
                Status::Failed => return Err(Error::Failed),
                Status::Committed => return Err(Error::Committed),
                Status::RolledBack => return Err(Error::RolledBack),
            }
            if entry.outputs.len() == self.capacity {
                return Err(Error::Saturated);
            }
            entry.status = Status::Running;
            state.owner = Owner::Busy;
        }
        Ok(Task {
            queue: self,
            preparation: Some(Operation {
                queue: self,
                armed: false,
            }),
            id: Some(id),
        })
    }

    /// Exclude preparation and other lifecycle operations, but not task execution.
    pub fn operation(&self) -> Result<Operation<'_, I, O>, Error> {
        {
            let mut state = self.state();
            state.ready()?;
            state.owner = Owner::Busy;
        }
        Ok(Operation {
            queue: self,
            armed: false,
        })
    }
}

/// One caller-owned task keeps its transaction Running until completion or drop.
/// This status describes admitted queue work, not collection locking.
/// Dropping without completion fails the transaction and preserves its output.
pub struct Task<'a, I: Ord, O> {
    queue: &'a TaskQueue<I, O>,
    preparation: Option<Operation<'a, I, O>>,
    id: Option<I>,
}

impl<I: Ord, O> Task<'_, I, O> {
    /// Record exactly once, before executing the selected operation.
    pub fn record(&mut self, output: O) -> Result<(), Error> {
        if self.preparation.is_none() {
            return Err(Error::Conflict);
        }
        {
            let mut state = self.queue.state();
            let entry = state.entry(self.id.as_ref().expect("active task"))?;
            entry.outputs.push(output);
        }
        self.preparation.take().expect("preparation").complete();
        Ok(())
    }

    /// A successful task must have recorded its output.
    pub fn complete(mut self) -> Result<(), Error> {
        if self.preparation.is_some() {
            return Err(Error::Failed);
        }
        {
            let mut state = self.queue.state();
            state.entry(self.id.as_ref().expect("active task"))?.status = Status::Active;
        }
        self.id.take();
        Ok(())
    }
}

impl<I: Ord, O> Drop for Task<'_, I, O> {
    fn drop(&mut self) {
        if let Some(id) = &self.id {
            // Lifecycle cannot remove an entry while this task owns execution.
            let mut state = self.queue.state();
            state.entries.get_mut(id).expect("active task").status = Status::Failed;
        }
    }
}

/// Exclusive lifecycle access. Arming requires explicit successful completion.
/// Cancellation cannot return an error, so interruption is latched for all clones.
/// The caller owns recovery or shutdown policy; the queue cannot repair external effects.
pub struct Operation<'a, I: Ord, O> {
    queue: &'a TaskQueue<I, O>,
    armed: bool,
}

impl<I: Ord, O> Operation<'_, I, O> {
    /// Arm before starting an external effect whose interruption requires reopening.
    pub fn arm(&mut self) {
        self.armed = true;
    }

    /// Seal successful outputs, retaining the decision until finalization.
    /// Unseen identities are admitted as read-only transactions.
    /// None is a duplicate commit; Some(empty) is a newly committed read-only transaction.
    /// Sealing arms the permit before removing outputs from the queue.
    pub fn commit(&mut self, id: &I) -> Result<Option<Vec<O>>, Error>
    where
        I: Clone,
    {
        let mut state = self.queue.state();
        let entry = state.admit(id.clone())?;
        match entry.status {
            Status::Committed => return Ok(None),
            Status::Running => return Err(Error::Busy),
            Status::Failed => return Err(Error::Failed),
            Status::RolledBack => return Err(Error::RolledBack),
            Status::Active => {}
        }
        self.armed = true;
        entry.status = Status::Committed;
        Ok(Some(std::mem::take(&mut entry.outputs)))
    }

    /// Validate rollback before the caller delegates its external effect.
    /// An unseen identity above the cutoff is eligible for rollback.
    pub fn check_rollback(&self, id: &I) -> Result<bool, Error> {
        let state = self.queue.state();
        state.check(id)?;
        match state.entries.get(id).map(|entry| entry.status) {
            Some(Status::Running) => Err(Error::Busy),
            Some(Status::Committed) => Err(Error::Committed),
            Some(Status::RolledBack) => Ok(false),
            _ => Ok(true),
        }
    }

    /// Discard outputs only after the caller's rollback succeeds.
    pub fn rollback(&mut self, id: &I) -> Result<(), Error>
    where
        I: Clone,
    {
        self.check_rollback(id)?;
        let mut state = self.queue.state();
        let entry = state.admit(id.clone())?;
        entry.status = Status::RolledBack;
        entry.outputs = Vec::new();
        Ok(())
    }

    /// Check covered tasks before the caller finalizes external state.
    pub fn check_finalize(&self, cutoff: &I) -> Result<(), Error> {
        self.queue
            .state()
            .entries
            .range(..=cutoff)
            .try_for_each(|(_, entry)| match entry.status {
                Status::Running => Err(Error::Busy),
                _ => Ok(()),
            })
    }

    /// Advance the cutoff after the caller's finalization succeeds; older cutoffs are no-ops.
    pub fn finalize(&mut self, cutoff: I) -> Result<(), Error> {
        self.check_finalize(&cutoff)?;
        let mut state = self.queue.state();
        if state
            .finalized
            .as_ref()
            .is_some_and(|prior| prior >= &cutoff)
        {
            return Ok(());
        }
        state.entries.retain(|id, _| id > &cutoff);
        state.finalized = Some(cutoff);
        Ok(())
    }

    #[cfg(test)]
    fn pending(&self) -> Vec<O>
    where
        O: Clone,
    {
        self.queue
            .state()
            .entries
            .values()
            .flat_map(|entry| entry.outputs.iter().cloned())
            .collect()
    }

    /// Acknowledge completion of all external effects before releasing exclusion.
    pub fn complete(mut self) {
        self.armed = false;
    }
}

impl<I: Ord, O> Drop for Operation<'_, I, O> {
    fn drop(&mut self) {
        self.queue.state().owner = if self.armed {
            Owner::Interrupted
        } else {
            Owner::Ready
        };
    }
}

#[cfg(test)]
mod tests {
    use super::TaskQueue;
    use crate::Error;

    #[test]
    fn admission_order_capacity_and_decisions() {
        let queue = TaskQueue::new(2);
        for output in [10, 20] {
            let mut task = queue.start(1).unwrap();
            assert_eq!(queue.operation().err(), Some(Error::Busy));
            assert_eq!(queue.start(2).err(), Some(Error::Busy));
            assert_eq!(queue.readable(1), Ok(()));
            queue.register(2).unwrap();
            task.record(output).unwrap();
            assert_eq!(queue.readable(1), Ok(()));
            assert_eq!(task.record(99), Err(Error::Conflict));
            task.complete().unwrap();
        }
        assert_eq!(queue.start(1).err(), Some(Error::Saturated));
        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.commit(&1).unwrap(), Some(vec![10, 20]));
        assert!(operation.pending().is_empty());
        operation.complete();

        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.commit(&1).unwrap(), None);
        assert_eq!(operation.check_rollback(&1), Err(Error::Committed));
        assert_eq!(operation.rollback(&1), Err(Error::Committed));
        operation.complete();
        assert_eq!(queue.start(1).err(), Some(Error::Committed));
        let mut task = queue.start(2).unwrap();
        task.record(30).unwrap();
        task.complete().unwrap();
    }

    #[test]
    fn failed_tasks_retain_outputs_until_rollback_or_cutoff() {
        let queue = TaskQueue::new(1);
        assert_eq!(queue.start(1).unwrap().complete(), Err(Error::Failed));
        assert_eq!(queue.readable(1), Err(Error::Failed));
        {
            let mut operation = queue.operation().unwrap();
            assert_eq!(operation.commit(&1), Err(Error::Failed));
            assert!(operation.pending().is_empty());
            assert!(operation.check_rollback(&1).unwrap());
            operation.rollback(&1).unwrap();
            assert!(!operation.check_rollback(&1).unwrap());
            operation.rollback(&1).unwrap();
            assert_eq!(operation.commit(&1), Err(Error::RolledBack));
        }
        assert_eq!(queue.readable(1), Err(Error::RolledBack));
        let mut task = queue.start(2).unwrap();
        task.record(42).unwrap();
        drop(task);
        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.pending(), vec![42]);
        assert_eq!(operation.commit(&2), Err(Error::Failed));
        operation.finalize(2).unwrap();
        operation.finalize(1).unwrap();
        operation.finalize(2).unwrap();
        assert!(operation.pending().is_empty());
        assert_eq!(operation.commit(&2), Err(Error::Outdated));
        assert_eq!(operation.rollback(&2), Err(Error::Outdated));
        operation.complete();
        assert_eq!(queue.register(1), Err(Error::Outdated));
        assert!(queue.start(3).is_ok());
    }

    #[test]
    fn finalization_preserves_future_outputs_and_read_only_receipts() {
        let queue = TaskQueue::<u64, u64>::new(1);
        queue.register(1).unwrap();
        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.commit(&1).unwrap(), Some(vec![]));
        operation.complete();
        let mut task = queue.start(3).unwrap();
        task.record(30).unwrap();
        task.complete().unwrap();
        let mut operation = queue.operation().unwrap();
        operation.finalize(2).unwrap();
        assert_eq!(operation.pending(), vec![30]);
        assert_eq!(operation.commit(&1), Err(Error::Outdated));
        assert_eq!(operation.commit(&3).unwrap(), Some(vec![30]));
        operation.complete();
    }

    #[test]
    fn execution_overlaps_without_blocking_other_transaction_decisions() {
        let queue = TaskQueue::new(2);
        let mut first = queue.start(1).unwrap();
        first.record(10).unwrap();
        assert_eq!(queue.start(1).err(), Some(Error::Busy));
        let mut later = queue.start(2).unwrap();
        later.record(30).unwrap();
        {
            let mut operation = queue.operation().unwrap();
            assert_eq!(operation.commit(&1), Err(Error::Busy));
            assert_eq!(operation.check_rollback(&1), Err(Error::Busy));
            assert_eq!(operation.check_finalize(&1), Err(Error::Busy));
            assert_eq!(operation.finalize(1), Err(Error::Busy));
            assert_eq!(operation.pending(), vec![10, 30]);
        }
        first.complete().unwrap();
        let mut second = queue.start(1).unwrap();
        second.record(20).unwrap();
        second.complete().unwrap();
        assert_eq!(queue.start(1).err(), Some(Error::Saturated));
        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.commit(&1).unwrap(), Some(vec![10, 20]));
        operation.finalize(1).unwrap();
        operation.complete();
        later.complete().unwrap();
        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.commit(&2).unwrap(), Some(vec![30]));
        operation.complete();
    }

    #[test]
    fn failure_does_not_exclude_other_transactions() {
        let queue = TaskQueue::new(2);
        let mut first = queue.start(1).unwrap();
        first.record(10).unwrap();
        let mut second = queue.start(2).unwrap();
        second.record(20).unwrap();
        drop(first);
        second.complete().unwrap();
        let mut operation = queue.operation().unwrap();
        assert_eq!(operation.commit(&1), Err(Error::Failed));
        assert_eq!(operation.pending(), vec![10, 20]);
        operation.rollback(&1).unwrap();
        assert_eq!(operation.commit(&2).unwrap(), Some(vec![20]));
        operation.complete();
        let mut independent = queue.start(3).unwrap();
        independent.record(30).unwrap();
        independent.complete().unwrap();
    }

    #[test]
    fn interruption_invalidates_all_clones_and_sealed_commits() {
        for sealing in [false, true] {
            let queue = TaskQueue::<u64, u64>::new(1);
            let clone = queue.clone();
            queue.register(1).unwrap();
            let mut live = queue.start(3).unwrap();
            live.record(30).unwrap();
            let mut operation = queue.operation().unwrap();
            if sealing {
                operation.commit(&1).unwrap();
            } else {
                operation.arm();
            }
            drop(operation);
            if sealing {
                assert_eq!(live.complete(), Err(Error::Interrupted));
            } else {
                drop(live);
            }
            assert_eq!(clone.register(2), Err(Error::Interrupted));
            assert_eq!(clone.readable(1), Err(Error::Interrupted));
            assert_eq!(clone.start(2).err(), Some(Error::Interrupted));
            assert_eq!(clone.operation().err(), Some(Error::Interrupted));
            assert!(clone.validate_fresh().is_err());
        }
    }

    #[test]
    fn rejected_decisions_do_not_interrupt_and_freshness_is_strict() {
        let queue = TaskQueue::<u64, u64>::new(1);
        queue.validate_fresh().unwrap();
        for id in 1..100 {
            queue.readable(id).unwrap();
        }
        queue.validate_fresh().unwrap();
        {
            let mut operation = queue.operation().unwrap();
            assert_eq!(operation.commit(&1), Ok(Some(vec![])));
            assert_eq!(operation.commit(&1), Ok(None));
            assert_eq!(operation.check_rollback(&1), Err(Error::Committed));
            assert_eq!(operation.check_rollback(&2), Ok(true));
            operation.rollback(&2).unwrap();
            assert_eq!(operation.check_rollback(&2), Ok(false));
            assert_eq!(operation.commit(&2), Err(Error::RolledBack));
            assert!(queue.validate_fresh().is_err());
            operation.complete();
        }
        assert_eq!(queue.readable(2), Err(Error::RolledBack));
        queue.register(1).unwrap();
        assert!(queue.validate_fresh().is_err());
        let mut operation = queue.operation().unwrap();
        operation.finalize(2).unwrap();
        operation.complete();
        assert_eq!(queue.readable(2), Err(Error::Outdated));
        assert_eq!(queue.start(2).err(), Some(Error::Outdated));
        {
            let operation = queue.operation().unwrap();
            assert_eq!(queue.start(2).err(), Some(Error::Busy));
            operation.complete();
        }
        assert_eq!(queue.readable(3), Ok(()));
        let mut task = queue.start(3).unwrap();
        task.record(7).unwrap();
        task.complete().unwrap();
        assert!(queue.validate_fresh().is_err());
    }

    #[tokio::test]
    async fn cancellation_drops_work_without_spawning_or_selecting_a_decision() {
        for recorded in [false, true] {
            let queue = TaskQueue::new(1);
            let completed = std::sync::atomic::AtomicBool::new(false);
            let mut future = Box::pin(async {
                let mut task = queue.start(1).unwrap();
                if recorded {
                    task.record(7).unwrap();
                }
                std::future::pending::<()>().await;
                completed.store(true, std::sync::atomic::Ordering::Relaxed);
                task.complete().unwrap();
            });
            assert!(futures::poll!(&mut future).is_pending());
            drop(future);
            tokio::task::yield_now().await;
            assert!(!completed.load(std::sync::atomic::Ordering::Relaxed));
            let mut operation = queue.operation().unwrap();
            assert_eq!(operation.pending().len(), usize::from(recorded));
            assert_eq!(operation.commit(&1), Err(Error::Failed));
            operation.rollback(&1).unwrap();
            assert!(operation.pending().is_empty());
            operation.complete();
            let mut task = queue.start(2).unwrap();
            task.record(8).unwrap();
            task.complete().unwrap();
        }
    }
}
