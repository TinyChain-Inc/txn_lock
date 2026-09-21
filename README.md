# txn_lock
A futures-aware read-write lock for Rust which supports transaction-specific versioning

`queue::task::TaskQueue<I, Out>` holds ordered outputs from caller-driven work.
Construct it with a positive per-transaction capacity, obtain a task with `start`,
record its output before executing the corresponding operation, and call
`complete` only after success. Dropping a task fails its transaction; no work is
spawned or detached. Callers separately bound the number of live transactions.

`record` releases preparation exclusion. Tasks in different transactions can
execute concurrently; earlier pending outputs do not exclude later transactions.
Each transaction keeps its own request sequence: starting another task while its
previous task is unfinished returns `Busy`. Admission changes its status from
`Active` to `Running`; completion restores `Active`, and unfinished drop marks
`Failed`. This is the authoritative queue status, not a collection lock or counter.

An exclusive operation permit provides fallible commit, rollback, and cutoff
finalization. Commit seals outputs and arms the permit; callers arm it explicitly
before other external effects. An armed permit must be completed, or all clones
become unusable. The caller owns recovery or shutdown. See the API documentation
in [`src/queue/task.rs`](src/queue/task.rs) for decision and cancellation semantics.
Commit and rollback require their transaction's tasks to finish. Call
`check_finalize(cutoff)` before external finalization; only covered tasks must
finish. Other transactions' execution does not exclude lifecycle operations.

Pending outputs include failed tasks until rollback or finalization discards them.
Committed and rolled-back decisions remain until cutoff finalization, making
duplicate decisions deterministic. The queue owns no storage or durability policy.

`readable(id)` validates an observation without registering it. Unseen identities
above the cutoff are readable; failed and rolled-back decisions are rejected.
Commit and rollback accept previously unseen identities and retain their decisions
until finalization. Explicit registration remains available.

`semaphore::Semaphore::try_resolve` protects a synchronous transaction decision
against live permits in that transaction, then releases its reservations on
success. It does not introduce a new write range or conflict with other
transactions' reservations. The decision callback must not reenter the semaphore
or perform I/O; errors leave reservations intact.

With sibling path dependencies available, run standalone checks from this directory:

```sh
cargo test --all-targets --all-features
cargo test --doc queue::task
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```
