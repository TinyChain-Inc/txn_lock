//! Utilities to support transactional versioning.
//!
//! General-purpose locks and usage examples are provided
//! in the [`map`], [`queue`], [`scalar`], and [`set`] modules.
//!
//! More complex transaction locks (e.g. for a relational database) can be constructed using
//! the [`semaphore`] module.

pub mod map;
pub mod queue;
pub mod scalar;
pub mod semaphore;
pub mod set;

mod guard;
mod range;

use std::fmt;

/// An error which may occur when attempting to acquire a transactional lock
#[derive(Clone, Eq, PartialEq)]
pub enum Error {
    /// Cannot acquire a write lock because the transaction is already committed
    Committed,

    /// Cannot write to the requested range because it's already been locked in the future
    Conflict,

    /// Cannot read or write to the requested range because it's already been finalized
    Outdated,

    /// Unable to acquire a transactional lock synchronously
    WouldBlock,

    /// A bounded transactional queue has reached its configured capacity
    Saturated,

    /// Another task or lifecycle operation holds exclusive admission.
    Busy,

    /// A task failed or was cancelled before completion.
    Failed,

    /// The transaction was rolled back.
    RolledBack,

    /// An armed operation did not complete; this owner cannot safely resume.
    Interrupted,

    /// An error occurred in a delegated operation.
    Background(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Committed => f.write_str("cannot acquire an exclusive lock after committing"),
            Self::Conflict => {
                f.write_str("there is a conflicting transactional lock on this resource")
            }
            Self::Outdated => f.write_str("the value has already been finalized"),
            Self::WouldBlock => f.write_str("synchronous lock acquisition failed"),
            Self::Saturated => f.write_str("transactional queue capacity exhausted"),
            Self::Busy => f.write_str("operation in progress"),
            Self::Failed => f.write_str("transaction contains a failed or cancelled task"),
            Self::RolledBack => f.write_str("transaction was rolled back"),
            Self::Interrupted => f.write_str("requires reload after an interrupted operation"),
            Self::Background(cause) => write!(f, "an error occured in a background task: {cause}"),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {}

impl From<tokio::sync::AcquireError> for Error {
    fn from(_: tokio::sync::AcquireError) -> Self {
        Self::Outdated
    }
}

impl From<tokio::sync::TryAcquireError> for Error {
    fn from(cause: tokio::sync::TryAcquireError) -> Self {
        match cause {
            tokio::sync::TryAcquireError::Closed => Error::Outdated,
            tokio::sync::TryAcquireError::NoPermits => Error::WouldBlock,
        }
    }
}

type Result<T> = std::result::Result<T, Error>;
