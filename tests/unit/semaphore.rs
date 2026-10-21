use std::cell::RefCell;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use collate::{Collator, OverlapsRange};
use futures::FutureExt;

use super::{Error, Semaphore};
use crate::set::Range as SetRange;

type Lock = Semaphore<u64, Collator<usize>, Range<usize>>;

fn lock() -> Lock {
    Lock::new(Collator::default())
}

#[test]
fn overlapping_reservations_exclude_writers_in_both_orders() {
    for (name, left, right) in [
        ("covering", 0..6, 3..4),
        ("partial", 0..4, 2..6),
        ("equal", 2..4, 2..4),
    ] {
        for (first, second) in [(left.clone(), right.clone()), (right, left)] {
            let semaphore = lock();
            let reader = semaphore.try_read(1, first.clone()).unwrap();
            let another = semaphore.try_read(1, second.clone()).unwrap();
            assert_eq!(
                semaphore.try_write(1, second.clone()).unwrap_err(),
                Error::WouldBlock,
                "{name}"
            );
            drop(another);
            assert_eq!(
                semaphore.try_write(1, second.clone()).unwrap_err(),
                Error::WouldBlock,
                "{name}"
            );
            drop(reader);

            let writer = semaphore.try_write(1, second.clone()).unwrap();
            assert_eq!(
                semaphore.try_read(1, first.clone()).unwrap_err(),
                Error::WouldBlock,
                "{name}"
            );
            assert_eq!(
                semaphore.try_write(1, first.clone()).unwrap_err(),
                Error::WouldBlock,
                "{name}"
            );
            drop(writer);
            semaphore.try_write(1, first).unwrap();
        }
    }
}

#[test]
fn disjoint_ranges_and_duplicates_keep_exact_granularity() {
    let semaphore = lock();
    let left = semaphore.try_write(1, 0..2).unwrap();
    let adjacent = semaphore.try_write(1, 2..4).unwrap();
    let right = semaphore.try_write(1, 6..8).unwrap();
    assert_eq!(semaphore.try_read(1, 1..7).unwrap_err(), Error::WouldBlock);
    drop((left, adjacent, right));
    let first = semaphore.try_read(1, 1..7).unwrap();
    let second = semaphore.try_read(1, 1..7).unwrap();
    assert_eq!(semaphore.versions.lock().unwrap()[&1].len(), 4);
    assert_eq!(*first, 1..7);
    drop((first, second));
    semaphore.try_write(1, 1..7).unwrap();
}

#[test]
fn concurrent_overlapping_attempts_grant_only_one_writer() {
    for (left, right) in [(0..6, 3..4), (0..4, 2..6), (2..4, 2..4)] {
        let semaphore = lock();
        let barrier = std::sync::Barrier::new(2);
        let acquired = std::thread::scope(|scope| {
            let attempt = |range| {
                barrier.wait();
                let permit = semaphore.try_write(1, range);
                barrier.wait();
                let acquired = permit.is_ok();
                drop(permit);
                acquired
            };
            let first = scope.spawn(move || attempt(left));
            let second = scope.spawn(move || attempt(right));
            usize::from(first.join().unwrap()) + usize::from(second.join().unwrap())
        });
        assert_eq!(acquired, 1);
        semaphore.try_write(1, 0..8).unwrap();
    }
}

#[test]
fn historical_ordering_rejections_do_not_register() {
    let semaphore = lock();
    drop(semaphore.try_write(2, 2..4).unwrap());
    assert_eq!(semaphore.try_read(3, 3..5).unwrap_err(), Error::WouldBlock);
    assert_eq!(semaphore.try_write(1, 3..5).unwrap_err(), Error::Conflict);
    assert_eq!(semaphore.versions.lock().unwrap().len(), 1);
    // The gap between two reservations is not itself reserved.
    drop(semaphore.try_read(2, 6..8).unwrap());
    semaphore.try_write(1, 4..6).unwrap();
    semaphore.finalize(&2, true);
    assert!(semaphore.versions.lock().unwrap().is_empty());
    semaphore.try_write(3, 3..5).unwrap();
}

#[test]
fn initial_and_contended_write_reservations_survive_permit_release() {
    let initial = Lock::with_reservation(1, Collator::default(), 0..8);
    assert_eq!(initial.try_read(2, 2..4).unwrap_err(), Error::WouldBlock);
    initial.try_read(1, 2..4).unwrap();
    initial.finalize(&1, false);
    initial.try_read(2, 2..4).unwrap();

    let semaphore = lock();
    let read = semaphore.try_read(1, 0..8).unwrap();
    assert_eq!(semaphore.try_write(1, 2..4).unwrap_err(), Error::WouldBlock);
    drop(read);
    assert_eq!(semaphore.try_read(2, 2..4).unwrap_err(), Error::WouldBlock);
    semaphore.try_read(2, 6..8).unwrap();
    semaphore.finalize(&1, false);
    semaphore.try_read(2, 2..4).unwrap();
}

#[tokio::test]
async fn async_waiters_retry_and_cancellation_releases_access() {
    let semaphore = lock();
    let read = semaphore.read(1, 0..8).await.unwrap();
    {
        let write = semaphore.write(1, 2..4);
        tokio::pin!(write);
        assert!(futures::poll!(&mut write).is_pending());
    }
    // Cancellation retains history but no active writer or orphaned waiter.
    assert_eq!(semaphore.try_read(2, 2..4).unwrap_err(), Error::WouldBlock);
    let write = semaphore.write(1, 2..4);
    tokio::pin!(write);
    assert!(futures::poll!(&mut write).is_pending());
    drop(read);
    let writer = tokio::time::timeout(Duration::from_secs(1), write)
        .await
        .unwrap()
        .unwrap();
    let waiting = semaphore.read(1, 0..8);
    tokio::pin!(waiting);
    assert!(futures::poll!(&mut waiting).is_pending());
    drop(writer);
    let reader = tokio::time::timeout(Duration::from_secs(1), waiting)
        .await
        .unwrap()
        .unwrap();
    drop(reader);

    let later = semaphore.read(2, 2..4);
    tokio::pin!(later);
    assert!(futures::poll!(&mut later).is_pending());
    semaphore.finalize(&1, false);
    tokio::time::timeout(Duration::from_secs(1), later)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn retry_rechecks_historical_conflicts_before_registration() {
    let semaphore = Lock::with_reservation(1, Collator::default(), 0..2);
    let waiting = semaphore.write(2, 0..6);
    tokio::pin!(waiting);
    assert!(futures::poll!(&mut waiting).is_pending());
    let later = semaphore.try_read(3, 4..6).unwrap();
    semaphore.finalize(&1, false);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap_err(),
        Error::Conflict
    );
    assert!(!semaphore.versions.lock().unwrap().contains_key(&2));
    drop(later);
}

#[tokio::test]
async fn notification_between_attempt_and_wait_is_retained() {
    let semaphore = lock();
    let reader = semaphore.try_read(1, 0..8).unwrap();
    let notified = semaphore.notify.notified();
    tokio::pin!(notified);
    notified.as_mut().enable();
    assert!(matches!(
        semaphore.attempt(1, Arc::new(2..4), true).unwrap(),
        super::Attempt::Pending(..)
    ));
    drop(reader);
    tokio::time::timeout(Duration::from_secs(1), notified)
        .await
        .unwrap();
    semaphore.try_write(1, 2..4).unwrap();
}

struct OnWake(Box<dyn Fn() + Send + Sync>);

impl Wake for OnWake {
    fn wake(self: Arc<Self>) {
        (self.0)();
    }
}

#[test]
fn notification_observes_released_permits_and_state_mutex() {
    for finalize in [false, true] {
        let semaphore = lock();
        let reader = semaphore.try_read(1, 0..8).unwrap();
        let observed = Arc::new(Mutex::new(None));
        let result = observed.clone();
        let clone = semaphore.clone();
        let waker = Waker::from(Arc::new(OnWake(Box::new(move || {
            let unlocked = clone.versions.try_lock().is_ok();
            // Avoid blocking a broken implementation inside its notification.
            let acquired = unlocked && clone.try_write(if finalize { 2 } else { 1 }, 2..4).is_ok();
            *result.lock().unwrap() = Some((unlocked, acquired));
        }))));
        if finalize {
            drop(reader);
            let mut notified = std::pin::pin!(semaphore.notify.notified());
            assert!(matches!(
                notified.poll_unpin(&mut Context::from_waker(&waker)),
                Poll::Pending
            ));
            semaphore.finalize(&1, false);
        } else {
            let mut notified = std::pin::pin!(semaphore.notify.notified());
            assert!(matches!(
                notified.poll_unpin(&mut Context::from_waker(&waker)),
                Poll::Pending
            ));
            drop(reader);
        }
        assert_eq!(*observed.lock().unwrap(), Some((true, true)));
    }
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Id(u64);

#[derive(Debug)]
struct NonCloneRange(Range<usize>);

impl collate::OverlapsRange<Self, Collator<usize>> for NonCloneRange {
    fn overlaps(&self, other: &Self, collator: &Collator<usize>) -> collate::Overlap {
        self.0.overlaps(&other.0, collator)
    }
}

#[tokio::test]
async fn acquisition_supports_non_clone_inputs_and_send_futures() {
    fn assert_send(_: &impl Send) {}
    let semaphore = Semaphore::new(Collator::<usize>::default());
    let reader = semaphore.try_read(Id(1), NonCloneRange(0..8)).unwrap();
    let future = semaphore.write(Id(1), NonCloneRange(2..4));
    assert_send(&future);
    tokio::pin!(future);
    assert!(futures::poll!(&mut future).is_pending());
    drop(reader);
    let permit = tokio::time::timeout(Duration::from_secs(1), future)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(permit.0, 2..4);
}

// Test-only injection at the acquisition-to-wait boundary.
tokio::task_local! {
    static BEFORE_WAIT: RefCell<Option<Box<dyn FnOnce()>>>;
}

pub(super) fn before_wait() {
    let _ = BEFORE_WAIT.try_with(|hook| {
        if let Some(hook) = hook.take() {
            hook();
        }
    });
}

#[test]
fn initial_reservation_blocks_overlapping_later_access() {
    let semaphore = Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::with_reservation(
        1,
        Collator::default(),
        2..4,
    );
    assert!(matches!(
        semaphore.try_read(2, 3..4),
        Err(Error::WouldBlock)
    ));
    assert!(matches!(
        semaphore.try_write(2, 3..4),
        Err(Error::WouldBlock)
    ));
    assert!(semaphore.try_read(2, 4..5).is_ok());
    assert!(semaphore.try_write(2, 5..6).is_ok());
    semaphore.finalize(&1, false);
    assert!(semaphore.try_read(2, 3..4).is_ok());
    assert!(semaphore.try_write(2, 3..4).is_ok());
}

#[tokio::test]
async fn finalization_between_reservation_check_and_wait_is_observed() {
    for write in [false, true] {
        let semaphore = Semaphore::<u64, Collator<u64>, SetRange<u64>>::new(Collator::default());
        drop(semaphore.try_write(1, SetRange::All).unwrap());
        let blocker = semaphore.clone();
        BEFORE_WAIT
            .scope(
                RefCell::new(Some(Box::new(move || blocker.finalize(&1, false)))),
                async {
                    let waiting = async {
                        if write {
                            drop(semaphore.write(2, SetRange::All).await.unwrap());
                        } else {
                            drop(semaphore.read(2, SetRange::All).await.unwrap());
                        }
                    };
                    futures::pin_mut!(waiting);
                    assert!(futures::poll!(waiting).is_ready());
                    assert!(BEFORE_WAIT.with(|hook| hook.borrow().is_none()));
                },
            )
            .await;
    }
}

#[tokio::test]
async fn cancelled_waiters_leave_no_later_reservation() {
    for write in [false, true] {
        let semaphore = Semaphore::<u64, Collator<u64>, SetRange<u64>>::new(Collator::default());
        drop(semaphore.try_write(1, SetRange::All).unwrap());
        {
            let waiting = async {
                if write {
                    drop(semaphore.write(2, SetRange::All).await.unwrap());
                } else {
                    drop(semaphore.read(2, SetRange::All).await.unwrap());
                }
            };
            futures::pin_mut!(waiting);
            assert!(futures::poll!(waiting).is_pending());
        }
        assert_eq!(semaphore.versions.lock().unwrap().len(), 1);
        semaphore.finalize(&1, false);
        assert!(semaphore.versions.lock().unwrap().is_empty());
        assert!(semaphore.try_write(3, SetRange::All).is_ok());
    }
}

#[tokio::test]
async fn covering_read_blocks_a_later_inserted_child_write() {
    let semaphore = Semaphore::<u64, Collator<u64>, SetRange<u64>>::new(Collator::default());
    let read = semaphore.try_read(1, SetRange::All).unwrap();
    assert!(matches!(
        semaphore.try_write(1, SetRange::One(Arc::new(2))),
        Err(Error::WouldBlock)
    ));
    assert!(tokio::time::timeout(
        std::time::Duration::from_millis(10),
        semaphore.write(1, SetRange::One(Arc::new(2)))
    )
    .await
    .is_err());
    drop(read);
    let write = semaphore.try_write(1, SetRange::One(Arc::new(2))).unwrap();
    assert!(matches!(
        semaphore.try_read(1, SetRange::All),
        Err(Error::WouldBlock)
    ));
    drop(write);
    assert!(semaphore.try_read(1, SetRange::All).is_ok());
}

#[test]
fn exact_ranges_preserve_overlap_exclusion_and_disjoint_independence() {
    let cases = [
        (4..8, 4..8, true),
        (0..20, 5..10, true),
        (5..10, 0..20, true),
        (0..10, 5..15, true),
        (5..15, 0..10, true),
        (0..5, 5..10, false),
        (10..20, 70..80, false),
        (70..80, 10..20, false),
    ];
    for (first, second, overlaps) in cases {
        for first_write in [false, true] {
            for second_write in [false, true] {
                let semaphore =
                    Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
                // Historical covering and intermediate ranges must not make
                // disjoint live operations conflict with each other.
                for history in [0..100, 40..60, 20..80] {
                    drop(semaphore.try_read(1, history).unwrap());
                }
                let held: Box<dyn std::fmt::Debug> = if first_write {
                    Box::new(semaphore.try_write(1, first.clone()).unwrap())
                } else {
                    Box::new(semaphore.try_read(1, first.clone()).unwrap())
                };
                let result = if second_write {
                    semaphore.try_write(1, second.clone()).map(drop)
                } else {
                    semaphore.try_read(1, second.clone()).map(drop)
                };
                assert_eq!(
                    result,
                    if overlaps && (first_write || second_write) {
                        Err(Error::WouldBlock)
                    } else {
                        Ok(())
                    },
                    "{first:?} write={first_write}, {second:?} write={second_write}"
                );
                drop(held);
                assert!(semaphore.try_write(1, second.clone()).is_ok());
            }
        }
    }
}

#[test]
fn ordering_checks_exact_history_including_partial_tails() {
    let semaphore = Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
    drop(semaphore.try_read(1, 0..100).unwrap());
    drop(semaphore.try_write(1, 10..20).unwrap());
    assert!(semaphore.try_read(2, 70..80).is_ok());
    assert!(matches!(
        semaphore.try_read(2, 15..16),
        Err(Error::WouldBlock)
    ));
    semaphore.finalize(&1, false);
    assert!(semaphore.try_read(2, 15..16).is_ok());

    let semaphore = Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
    for range in [0..5, 10..15, 3..12] {
        drop(semaphore.try_read(3, range).unwrap());
    }
    for tail in [0..1, 14..15] {
        assert!(matches!(semaphore.try_write(2, tail), Err(Error::Conflict)));
        assert!(!semaphore.versions.lock().unwrap().contains_key(&2));
    }
    assert!(semaphore.try_write(2, 15..16).is_ok());
}

#[tokio::test]
async fn blocked_same_transaction_attempts_retain_history_without_active_permits() {
    for asynchronous in [false, true] {
        let semaphore =
            Semaphore::<u64, Collator<u64>, std::ops::Range<u64>>::new(Collator::default());
        let read = semaphore.try_read(1, 0..5).unwrap();
        if asynchronous {
            let waiting = semaphore.write(1, 0..20);
            futures::pin_mut!(waiting);
            assert!(futures::poll!(waiting).is_pending());
            // Dropping the future cancels waiting, not its range history.
        } else {
            assert!(matches!(
                semaphore.try_write(1, 0..20),
                Err(Error::WouldBlock)
            ));
        }
        drop(read);
        assert!(matches!(
            semaphore.try_read(2, 10..15),
            Err(Error::WouldBlock)
        ));
        assert!(semaphore.try_write(1, 10..15).is_ok());
        semaphore.finalize(&1, false);
        assert!(semaphore.try_read(2, 10..15).is_ok());
    }
}

#[tokio::test]
async fn permit_release_between_attempt_and_wait_is_observed() {
    for write in [false, true] {
        let semaphore = Semaphore::<u64, Collator<u64>, SetRange<u64>>::new(Collator::default());
        let held = semaphore.try_write(1, SetRange::All).unwrap();
        BEFORE_WAIT
            .scope(RefCell::new(Some(Box::new(move || drop(held)))), async {
                let waiting = async {
                    if write {
                        drop(semaphore.write(1, SetRange::All).await.unwrap());
                    } else {
                        drop(semaphore.read(1, SetRange::All).await.unwrap());
                    }
                };
                futures::pin_mut!(waiting);
                assert!(futures::poll!(waiting).is_ready());
                assert!(BEFORE_WAIT.with(|hook| hook.borrow().is_none()));
            })
            .await;
    }
}

#[tokio::test]
async fn notification_observes_released_permits_and_unlocked_history() {
    use futures::task::{waker, ArcWake};
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::task::Context;

    struct Observer {
        semaphore: Semaphore<u64, Collator<u64>, SetRange<u64>>,
        txn_id: u64,
        ready: AtomicBool,
    }
    impl ArcWake for Observer {
        fn wake_by_ref(this: &Arc<Self>) {
            // A custom waker may run immediately inside notify_waiters.
            // Never hang the regression if it observes the old lock order.
            let unlocked = this.semaphore.versions.try_lock().is_ok();
            let ready = unlocked && this.semaphore.try_write(this.txn_id, SetRange::All).is_ok();
            this.ready.store(ready, AtomicOrdering::SeqCst);
        }
    }

    for finalize in [false, true] {
        let semaphore = Semaphore::<u64, Collator<u64>, SetRange<u64>>::new(Collator::default());
        let held = semaphore.try_write(1, SetRange::All).unwrap();
        let txn_id = if finalize { 2 } else { 1 };
        let observer = Arc::new(Observer {
            semaphore: semaphore.clone(),
            txn_id,
            ready: AtomicBool::new(false),
        });
        let wake = waker(observer.clone());
        let mut context = Context::from_waker(&wake);
        let waiting = semaphore.read(txn_id, SetRange::All);
        futures::pin_mut!(waiting);
        assert!(waiting.as_mut().poll(&mut context).is_pending());
        if finalize {
            drop(held);
            // Re-poll after permit release: history still blocks txn 2.
            assert!(waiting.as_mut().poll(&mut context).is_pending());
            observer.ready.store(false, AtomicOrdering::SeqCst);
            semaphore.finalize(&1, false);
        } else {
            drop(held);
        }
        assert!(observer.ready.load(AtomicOrdering::SeqCst));
        assert!(waiting.as_mut().poll(&mut context).is_ready());
    }
}

#[tokio::test]
async fn acquisition_requires_no_clone_bound_on_ids_or_ranges() {
    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
    struct Id(u64);
    #[derive(Debug)]
    struct Selection(std::ops::Range<u64>);
    impl OverlapsRange<Self, Collator<u64>> for Selection {
        fn overlaps(&self, other: &Self, collator: &Collator<u64>) -> collate::Overlap {
            self.0.overlaps(&other.0, collator)
        }
    }

    let semaphore = Semaphore::<Id, Collator<u64>, Selection>::new(Collator::default());
    let read = semaphore.try_read(Id(1), Selection(2..4)).unwrap();
    let waiting = semaphore.write(Id(1), Selection(2..4));
    futures::pin_mut!(waiting);
    assert!(futures::poll!(waiting.as_mut()).is_pending());
    drop(read);
    let write = waiting.await.unwrap();
    assert_eq!(write.0, 2..4);
}

#[test]
fn equal_ranges_share_one_reservation_and_preserve_write_history() {
    let semaphore = Semaphore::<u64, Collator<u64>, Range<u64>>::new(Collator::default());
    let read = semaphore.try_read(1, 2..4).unwrap();
    for _ in 0..64 {
        drop(semaphore.try_read(1, 2..4).unwrap());
        assert!(matches!(
            semaphore.try_write(1, 2..4),
            Err(Error::WouldBlock)
        ));
    }
    {
        let versions = semaphore.versions.lock().unwrap();
        assert_eq!(versions[&1].len(), 1);
        assert!(versions[&1][0].write);
    }
    drop(read);
    drop(semaphore.try_write(1, 2..4).unwrap());
    assert!(matches!(
        semaphore.try_read(2, 2..4),
        Err(Error::WouldBlock)
    ));
}
