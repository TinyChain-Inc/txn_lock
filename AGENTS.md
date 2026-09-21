# txn_lock Agent Notes

- Transaction ordering semaphores are consistency capabilities, not work queues.
  Preserve canonical async wait/wake behavior and return `WouldBlock` only from
  explicitly synchronous `try_*` or bounded-admission APIs.
- Every message/task queue requires an explicit positive capacity. Reject at the
  limit before spawning work, and release capacity on commit, rollback, finalize,
  cancellation, or drop. Do not add unbounded channels, vectors, or task sets.
- A queue bound applies per transaction; callers separately bound admitted live
  transactions. Do not duplicate caller-level admission here.
- Canonical state may initialize an unpublished lock. Once published, every
  observation requires the caller's transaction ID and the corresponding
  transactional read API.
