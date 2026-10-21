use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use collate::Collator;
use futures::FutureExt;

use super::{Error, Semaphore};

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
