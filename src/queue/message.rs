//! A transactional message queue.
//!
//! Example:
//! ```
//! use txn_lock::queue::message::MessageQueue;
//! use txn_lock::Error;
//!
//! let queue = MessageQueue::<u64, &'static str>::new(16);
//!
//! assert_eq!(queue.commit(0).unwrap(), Vec::<&'static str>::new());
//!
//! queue.push(1, "first message").expect("send");
//! queue.push(2, "second message").expect("send");
//! queue.push(3, "third message").expect("send");
//!
//! assert_eq!(queue.commit(2).unwrap(), vec!["second message"]);
//! assert_eq!(queue.commit(1).unwrap(), vec!["first message"]);
//!
//! queue.finalize(2).unwrap();
//!
//! assert_eq!(queue.commit(3).unwrap(), vec!["third message"]);
//!
//! queue.push(4, "fourth message").expect("send");
//! ```

use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use super::{Entry, State};
use crate::Error;

/// A transactional message queue
pub struct MessageQueue<I, M> {
    capacity: usize,
    state: Arc<Mutex<State<I, Vec<M>>>>,
}

impl<I, M> Clone for MessageQueue<I, M> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            state: self.state.clone(),
        }
    }
}

impl<I, M> MessageQueue<I, M> {
    /// Construct a new [`MessageQueue`].
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "message queue capacity must be positive");

        Self {
            capacity,
            state: Arc::new(Mutex::new(State::new())),
        }
    }
}

impl<I: Copy + Eq + Ord + Hash, M> MessageQueue<I, M> {
    /// Push the given `message` onto the queue at `txn_id`.
    pub fn push(&self, txn_id: I, message: M) -> Result<(), Error> {
        let mut state = self.state.lock().expect("transaction bookkeeping poisoned");

        match state.check_pending(txn_id)? {
            Entry::Occupied(mut entry) if entry.get().len() < self.capacity => {
                entry.get_mut().push(message)
            }
            Entry::Occupied(_) => return Err(Error::Saturated),
            Entry::Vacant(entry) => {
                entry.insert(vec![message]);
            }
        };

        Ok(())
    }
}

impl<I: Eq + Hash + Ord + fmt::Debug, M> MessageQueue<I, M> {
    /// Close and return the queue for the given `txn_id`, if any.
    ///
    /// If `commit` is called multiple times, only the first will return a filled `Vec`.
    ///
    /// Returns an error for an invalid decision.
    pub fn commit(&self, txn_id: I) -> Result<Vec<M>, Error> {
        let mut state = self.state.lock().expect("transaction bookkeeping poisoned");
        Ok(state.commit(txn_id)?.unwrap_or_default())
    }

    /// Close the queue at `txn_id`, if any.
    ///
    /// Returns an error for an invalid decision.
    pub fn rollback(&self, txn_id: &I) -> Result<(), Error> {
        let mut state = self.state.lock().expect("transaction bookkeeping poisoned");
        state.rollback(txn_id)?;
        Ok(())
    }

    /// Finalize this [`MessageQueue`], preventing any further actions prior to the given `txn_id`.
    pub fn finalize(&self, txn_id: I) -> Result<(), Error>
    where
        I: PartialOrd,
    {
        let mut state = self.state.lock().expect("transaction bookkeeping poisoned");
        state.finalize(txn_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MessageQueue;
    use crate::Error;

    #[test]
    fn applies_backpressure_at_capacity() {
        let queue = MessageQueue::new(1);

        queue.push(1, "first").expect("first message");
        assert_eq!(queue.push(1, "second"), Err(Error::Saturated));
        assert_eq!(queue.commit(1).unwrap(), vec!["first"]);
    }
}
