# txn_lock
A futures-aware read-write lock for Rust which supports transaction-specific versioning

`TxnMapLock::inspect_versions` synchronously inspects retained baseline,
committed, and pending entries, including removals. Its callback cannot retain
borrowed values or reenter the lock; a busy pending value returns `WouldBlock`,
possibly after earlier callbacks. It grants no transactional read capability.
Storage owners must coordinate native publication separately when using this
inspection for cutoff cleanup.
`TxnMapLock::is_committed` inspects existing transaction decisions without
admission or mutation; it treats the exact finalized cutoff as committed and
rejects older IDs. Callers synchronize external publication separately.

`TxnMapLock::inspect_committed` inspects one key's finalized baseline and latest
committed value without reading pending values. A committed removal remains
absence, and borrowed keys need no allocation. Its callback has the same
bookkeeping and publication restrictions; IDs below the cutoff reject.

Inspection requires a cutoff ID and rejects outdated cutoffs. `TxnLock::ptr_eq`
compares shared-owner identity without reading values, allowing lifecycle callers
to avoid processing clones of the same owner repeatedly.

`queue::task::TaskQueue<I, Out>` holds ordered outputs from caller-driven work.
Construct it with a positive per-transaction capacity, await a task with `start`,
record its output before executing the corresponding operation, and call
`complete` only after success. Dropping a task fails its transaction; no work is
spawned or detached. Callers separately bound the number of live transactions.

`record` releases preparation exclusion. Tasks in different transactions can
execute concurrently; earlier pending outputs do not exclude later transactions.
Each transaction keeps its own request sequence: starting another task while its
previous task is unfinished returns `Busy`. Admission changes its status from
`Active` to `Running`; completion restores `Active`, and unfinished drop marks
`Failed`. This is the authoritative queue status, not a collection lock or counter.

Awaiting `operation` provides exclusive access for fallible commit, rollback, and cutoff
finalization. Commit seals outputs and arms the permit; callers arm it explicitly
before other external effects. An armed permit must be completed, or all clones
become unusable. A single semaphore owns exclusion; interruption closes it and
wakes waiting callers with an error. Waiting for admission registers no work,
and cancellation while waiting selects no decision. Callers bound waiting work
and enforce their original deadline. The caller owns recovery or shutdown. See the API documentation
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

Transaction-ID histories use standard `BTreeMap` and `BTreeSet` indexes.
Visibility uses bounded chronological ranges, and cutoff cleanup removes native
first entries without shifting vector indexes. Keys are owned directly; these
indexes do not require shared key owners. Independently retained values and
transaction guards still keep their existing shared ownership. Exact lookup is
logarithmic tree lookup rather than hash lookup; this is not a throughput claim.

Scalar, map, set, and message-queue lifecycle methods return `Result`.
Outdated or conflicting decisions return their existing error variants before
changing versions or releasing reservations. Scalar/map commit also returns
`Busy` while a pending value is borrowed. This does not replace the caller's
quiescence obligation. Poisoned bookkeeping panics; callers must not inspect or
resume poisoned contents. `Interrupted` still identifies an armed operation which
did not complete and requires the caller to recover its owner.
Infallible iterators already acquired before a caller panic must not be reused.
Queue constructors require positive capacity as a trusted native precondition.
The library neither chooses distributed decisions nor terminates the process.

Map and set `read_and_commit` return changes only when that call first commits
the version. Repeated decisions return the historical snapshot with `None`
changes. Reads remain supported at the finalized cutoff and reject earlier IDs.
Map `check_commit` validates decision eligibility without admission or mutation;
it does not reserve an outcome. `try_read_and_commit` acquires causal admission
synchronously and returns `WouldBlock` when it cannot proceed. It shares the
asynchronous operation's decision and repeated-result semantics. Storage owners
can admit a full-map read before native work, then decide without awaiting while
holding publication exclusion. They must still finish and release pending-value
borrows before deciding. Rejected synchronous set mutations register no reservation.
Plain scalar, map, and set commits also succeed idempotently at the exact
finalized cutoff. Strictly older decisions remain outdated.

Map `read_and_rollback` returns discarded membership: the version that would
have been committed. Values remain shared without payload cloning. Restored state
is available through ordinary transactional reads; no second snapshot is returned.
Repeated rollback and exact-cutoff calls return the current baseline membership.
Vacant entries can inspect the committed value hidden by a removal, so resource
owners can preserve identity when recreating a name within the same transaction.
`EntryVacant::get_committed` rejects newer committed changes to that exact key,
including deletion, with `Conflict`. It retains no extra history or values;
ordinary historical reads and unrelated-key changes remain supported.

`semaphore::Semaphore::finalize` releases transaction reservations after the
resource's decision succeeds. The caller first finishes the affected operations
and releases their permits; the semaphore does not enforce lifecycle quiescence.

Within each transaction, the semaphore retains one exact reservation per equal
range until finalization. Active overlap checks scan these reservations under the
same mutex as registration and acquire one native permit. The mutex never spans
caller I/O. This avoids cloned range topology and permits disjoint operations
to proceed independently. Work is linear in the transaction's retained range
history; no range index or waiter queue is maintained. Async callers retry on notification without FIFO ordering or a
starvation-freedom guarantee. Callers retain their existing request deadlines.

A blocked same-transaction attempt retains its historical reservation even if
cancelled. An attempt waiting on an earlier transaction, or rejected by conflicting
future history, creates no reservation. Dropping active permits frees capacity
before waking waiters; it does not erase transaction history.

With sibling path dependencies available, run standalone checks from this directory:

```sh
cargo test --all-targets --all-features
cargo test --doc queue::task
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

Iterator values which borrow a pending native version retain the transaction
permit after their iterator is dropped. Mutable iterator values do the same for
exclusive access. Their public guard types carry both key and value type
parameters; native access is released before the final shared permit notifies
waiters. Committed immutable values retain only their existing payload ownership.
