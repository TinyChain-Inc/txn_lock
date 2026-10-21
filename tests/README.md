# Range-lock regression ownership

Private semaphore tests live in `unit/semaphore.rs`. They exercise one shared
acquisition implementation; the task-local before-wait hook exists only in tests.

The reconciliation retains all ten semaphore tests from each branch with their
names unchanged. Main's former
`semaphore::version::tests::equal_ranges_share_one_reservation_and_preserve_write_history`
is now `semaphore::tests::equal_ranges_share_one_reservation_and_preserve_write_history`.
Its 64 duplicate acquisitions, single-reservation/write-history assertions, and
post-release acquisition remain; historical blocking is checked through the
public semaphore instead of the removed `Version` helper.

| Coverage | Retained fixtures |
|---|---|
| Exact overlap and disjoint access | Both-order range tests and main's read/write combinations with covering historical reservations |
| Historical ordering | Initial reservations, partial tails, rejection before registration, and conflict rechecking after wakeup |
| Cancellation | Same-transaction history retention and absence of reservations while waiting on earlier transactions |
| Notification ordering | Direct notification registration, public acquisition boundary injection, and immediate wakers checking released permits and history mutex |
| Generic inputs | Both non-Clone input cases; the fix branch additionally checks Send futures |

Run `cargo test --lib semaphore::tests` for focused coverage, and `cargo test`
and `cargo test --all-features` for native suites and doctests. Collection-level
Table/Tensor tests remain with `tc-collection`; they verify range mapping and
retained stream permits rather than reimplementing these overlap cases.
