use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::map::{Entry, TxnMapLock};
use crate::queue::message::MessageQueue;
use crate::scalar::TxnLock;
use crate::set::TxnSetLock;
use crate::Error;

#[tokio::test]
async fn synchronous_map_commit_preserves_admission_and_decision_checks() {
    let map = TxnMapLock::from_committed([(1, 10)]);
    map.insert(2, 1, 20).await.unwrap();
    map.insert(3, 2, 30).await.unwrap();
    assert_eq!(map.is_committed(2), Ok(false));
    assert_eq!(map.is_committed(3), Ok(false));
    assert_eq!(map.check_commit(3), Ok(()));
    assert_eq!(map.try_read_and_commit(3).unwrap_err(), Error::WouldBlock);
    assert_eq!(map.try_insert(4, 1, 40).unwrap_err(), Error::WouldBlock);
    assert_eq!(*map.get(2, &1).await.unwrap().unwrap(), 20);

    let (earlier, changes) = map.try_read_and_commit(2).unwrap();
    assert_eq!(map.is_committed(2), Ok(true));
    assert_eq!(map.is_committed(3), Ok(false));
    assert_eq!(*earlier.get(&1).unwrap().as_ref(), 20);
    assert_eq!(changes.unwrap().len(), 1);
    let borrowed = map.get(3, &2).await.unwrap().unwrap();
    assert_eq!(map.check_commit(3), Err(Error::Busy));
    assert_eq!(map.try_read_and_commit(3).unwrap_err(), Error::Busy);
    // A rejected decision must not add an all-keys read reservation.
    assert!(map.try_insert(4, 99, 99).unwrap().is_none());
    assert_eq!(map.try_insert(4, 2, 40).unwrap_err(), Error::WouldBlock);
    assert_eq!(*borrowed, 30);
    drop(borrowed);

    assert_eq!(map.check_commit(3), Ok(()));
    let (contents, changes) = map.try_read_and_commit(3).unwrap();
    assert_eq!(map.is_committed(3), Ok(true));
    assert_eq!(contents.len(), 2);
    assert_eq!(*contents.get(&1).unwrap().as_ref(), 20);
    assert_eq!(*contents.get(&2).unwrap().as_ref(), 30);
    assert_eq!(changes.unwrap().len(), 1);
    assert_eq!(
        map.try_read_and_commit(3).unwrap(),
        (contents.clone(), None)
    );

    assert_eq!(
        map.read_and_commit(3).await.unwrap(),
        (contents.clone(), None)
    );

    map.insert(4, 2, 40).await.unwrap();
    map.finalize(3).unwrap();
    assert_eq!(map.is_committed(3), Ok(true));
    assert_eq!(map.is_committed(2), Err(Error::Outdated));
    assert_eq!(map.is_committed(4), Ok(false));
    assert_eq!(map.check_commit(3), Ok(()));
    assert_eq!(map.try_read_and_commit(3).unwrap(), (contents, None));
    assert_eq!(map.check_commit(2), Err(Error::Outdated));
    assert_eq!(map.try_read_and_commit(2).unwrap_err(), Error::Outdated);
    assert_eq!(*map.get(4, &2).await.unwrap().unwrap(), 40);
    map.try_read_and_commit(4).unwrap();
}

#[tokio::test]
async fn retained_map_inspection_covers_versions_and_busy_values() {
    let map = TxnMapLock::from_committed([(1, 10), (2, 20)]);
    map.insert(2, 1, 11).await.unwrap();
    map.remove(2, &2).await.unwrap();
    map.commit(2).unwrap();
    map.insert(3, 3, 30).await.unwrap();
    let committed = |txn_id, key| {
        map.inspect_committed(txn_id, key, |baseline, latest, cutoff| {
            (baseline.copied(), latest.copied(), cutoff)
        })
    };

    assert_eq!(committed(1, &1), Ok((Some(10), Some(11), None)));
    assert_eq!(committed(2, &2), Ok((Some(20), None, None)));
    assert_eq!(committed(3, &3), Ok((None, None, None)));
    assert_eq!(committed(3, &99), Ok((None, None, None)));

    let collect = || {
        let mut entries = BTreeSet::new();
        map.inspect_versions(2, |id, key, value| {
            entries.insert((id, *key, value.copied()));
        })
        .unwrap();

        entries
    };

    assert_eq!(
        collect(),
        BTreeSet::from([
            (None, 1, Some(10)),
            (None, 2, Some(20)),
            (Some(2), 1, Some(11)),
            (Some(2), 2, None),
            (Some(3), 3, Some(30)),
        ])
    );

    let guard = map.get_mut(3, 3).await.unwrap().unwrap();
    assert_eq!(
        map.inspect_versions(2, |_, _, _| {}).unwrap_err(),
        Error::WouldBlock
    );

    // Pending locks do not prevent committed-only metadata inspection.
    assert_eq!(committed(3, &3), Ok((None, None, None)));
    drop(guard);
    map.commit(3).unwrap();
    assert_eq!(committed(2, &3), Ok((None, Some(30), None)));
    map.finalize(2).unwrap();
    assert_eq!(committed(2, &1), Ok((Some(11), Some(11), Some(2))));
    assert_eq!(committed(2, &2), Ok((None, None, Some(2))));
    assert_eq!(committed(2, &3), Ok((None, Some(30), Some(2))));
    assert_eq!(
        map.inspect_committed(1, &1, |_, _, _| panic!("outdated inspection")),
        Err::<(), _>(Error::Outdated)
    );

    assert_eq!(
        map.inspect_versions(1, |_, _, _| {}).unwrap_err(),
        Error::Outdated
    );

    assert_eq!(
        collect(),
        BTreeSet::from([(None, 1, Some(11)), (Some(3), 3, Some(30))])
    );

    let text = TxnMapLock::from_committed([("name".to_owned(), 7)]);
    assert_eq!(
        text.inspect_committed(1, "name", |baseline, latest, cutoff| {
            (baseline.copied(), latest.copied(), cutoff)
        }),
        Ok((Some(7), Some(7), None))
    );
}

#[tokio::test]
async fn rollback_returns_discarded_membership_and_restores_state() {
    #[derive(Debug)]
    struct Value(u64);

    let map = TxnMapLock::with_contents(1, [(1, Value(10)), (2, Value(20)), (3, Value(30))]);
    map.commit(1).unwrap();
    assert_eq!(
        map.read_and_rollback(1).await.unwrap_err(),
        Error::Committed
    );

    drop(map.remove(2, &1).await.unwrap());
    map.insert(2, 2, Value(200)).await.unwrap();
    map.insert(2, 4, Value(40)).await.unwrap();
    let guard = map.get(2, &2).await.unwrap().unwrap();
    assert_eq!(map.read_and_rollback(2).await.unwrap_err(), Error::Busy);
    assert_eq!(guard.0, 200);
    drop(guard);

    let discarded = map.read_and_rollback(2).await.unwrap();
    let restored = map
        .iter(2)
        .await
        .unwrap()
        .map(|(key, value)| (*key, value.0))
        .collect::<BTreeMap<_, _>>();

    let values =
        |snapshot: &std::collections::HashMap<crate::map::Key<i32>, std::sync::Arc<Value>>| {
            snapshot
                .iter()
                .map(|(key, value)| (**key, value.0))
                .collect::<BTreeMap<_, _>>()
        };

    assert_eq!(restored, BTreeMap::from([(1, 10), (2, 20), (3, 30)]));
    assert_eq!(
        values(&discarded),
        BTreeMap::from([(2, 200), (3, 30), (4, 40)])
    );

    let unchanged = map.get(2, &3).await.unwrap().unwrap();
    assert!(std::ptr::eq(&*unchanged, &**discarded.get(&3).unwrap()));
    drop(unchanged);
    map.insert(3, 1, Value(11)).await.unwrap();
    let again = map.read_and_rollback(2).await.unwrap();
    assert_eq!(values(&again), restored);
    assert_eq!(map.get(3, &1).await.unwrap().unwrap().0, 11);
    map.finalize(2).unwrap();
    let at_cutoff = map.read_and_rollback(2).await.unwrap();
    assert_eq!(values(&at_cutoff), restored);
    assert_eq!(map.read_and_rollback(1).await.unwrap_err(), Error::Outdated);
}

#[tokio::test]
async fn committed_inspection_tracks_baseline_and_latest_without_pending_values() {
    let lock = TxnLock::new(0);
    assert!(lock.ptr_eq(&lock.clone()));
    assert!(!lock.ptr_eq(&TxnLock::new(0)));
    let guard = lock.read(1).await.unwrap();
    assert_eq!(
        lock.inspect_committed(1, |base, latest, cutoff| (*base, *latest, cutoff)),
        Ok((0, 0, None))
    );

    drop(guard);
    lock.commit(1).unwrap();

    *lock.write(2).await.unwrap() = 2;
    let guard = lock.read(2).await.unwrap();
    assert_eq!(
        lock.inspect_committed(2, |base, latest, cutoff| (*base, *latest, cutoff)),
        Ok((0, 0, None))
    );

    drop(guard);
    lock.commit(2).unwrap();
    let guard = lock.read(3).await.unwrap();
    assert_eq!(
        lock.inspect_committed(3, |base, latest, cutoff| (*base, *latest, cutoff)),
        Ok((0, 2, None))
    );

    drop(guard);
    lock.rollback(&3).unwrap();

    lock.finalize(2).unwrap();
    let guard = lock.read(2).await.unwrap();
    assert_eq!(
        lock.inspect_committed(2, |base, latest, cutoff| (*base, *latest, cutoff)),
        Ok((2, 2, Some(2)))
    );

    assert_eq!(
        lock.inspect_committed(1, |_, _, _| panic!("outdated inspection")),
        Err::<(), _>(Error::Outdated)
    );

    drop(guard);
}

#[tokio::test]
async fn busy_scalar_decisions_preserve_values_and_reservations() {
    let lock = TxnLock::new(0);
    let mut value = lock.write(1).await.unwrap();
    *value = 7;
    assert_eq!(lock.commit(1), Err(Error::Busy));
    assert_eq!(*value, 7);
    assert_eq!(lock.try_write(2).unwrap_err(), Error::WouldBlock);
    drop(value);

    let value = lock.read(1).await.unwrap();
    assert_eq!(lock.read_and_rollback(1).await.unwrap_err(), Error::Busy);
    assert_eq!(*value, 7);
    assert_eq!(lock.try_write(2).unwrap_err(), Error::WouldBlock);
    drop(value);

    lock.commit(1).unwrap();
    lock.commit(1).unwrap();
    assert_eq!(lock.rollback(&1), Err(Error::Committed));
    assert_eq!(*lock.read(2).await.unwrap(), 7);
    lock.finalize(2).unwrap();
    assert_eq!(lock.commit(1), Err(Error::Outdated));
    assert_eq!(lock.rollback(&1), Err(Error::Outdated));
    assert_eq!(lock.read_and_commit(1).await.unwrap_err(), Error::Outdated);

    let history = TxnLock::new(10);
    for (txn_id, expected) in [(2, 20), (4, 40), (6, 60)] {
        *history.write(txn_id).await.unwrap() = expected;
        history.commit(txn_id).unwrap();
    }

    history.commit(4).unwrap();
    *history.write(8).await.unwrap() = 80;

    for (txn_id, expected) in [
        (0, 10),
        (1, 10),
        (2, 20),
        (3, 20),
        (4, 40),
        (5, 40),
        (6, 60),
        (7, 60),
        (8, 80),
    ] {
        assert_eq!(*history.read(txn_id).await.unwrap(), expected);
    }

    history.finalize(4).unwrap();
    assert_eq!(history.try_read(3).unwrap_err(), Error::Outdated);
    for (txn_id, expected) in [(4, 40), (5, 40), (6, 60), (7, 60), (8, 80)] {
        assert_eq!(*history.read(txn_id).await.unwrap(), expected);
    }

    assert_eq!(history.rollback(&6), Err(Error::Committed));
    history.finalize(7).unwrap();
    assert_eq!(*history.read(7).await.unwrap(), 60);
    assert_eq!(*history.read(8).await.unwrap(), 80);
    history.finalize(9).unwrap();
    history.finalize(9).unwrap();
    assert_eq!(*history.read(9).await.unwrap(), 60);
    assert_eq!(history.commit(8), Err(Error::Outdated));
    *history.write(10).await.unwrap() = 100;
    history.commit(10).unwrap();
    assert_eq!(*history.read(11).await.unwrap(), 100);
}

#[tokio::test]
async fn busy_map_decisions_leave_the_entire_pending_version_intact() {
    let map = TxnMapLock::with_contents(1, [(1, 10), (2, 20)]);
    let value = map.get(1, &2).await.unwrap().unwrap();
    assert_eq!(map.commit(1), Err(Error::Busy));
    assert_eq!(map.read_and_rollback(1).await.unwrap_err(), Error::Busy);
    assert_eq!(map.try_insert(2, 3, 30).unwrap_err(), Error::WouldBlock);
    assert_eq!(*value, 20);
    drop(value);
    assert_eq!(*map.get(1, &1).await.unwrap().unwrap(), 10);
    let (contents, changes) = map.read_and_commit(1).await.unwrap();
    assert_eq!(contents.len(), 2);
    assert_eq!(changes.unwrap().len(), 2);
    let (repeated, changes) = map.read_and_commit(1).await.unwrap();
    assert_eq!(repeated, contents);
    assert!(changes.is_none());
    map.commit(1).unwrap();
    assert_eq!(map.rollback(&1), Err(Error::Committed));
    map.finalize(2).unwrap();
    assert_eq!(map.commit(1), Err(Error::Outdated));
    assert_eq!(map.read_and_rollback(1).await.unwrap_err(), Error::Outdated);

    let history = TxnMapLock::from_committed([(1, 10), (2, 20)]);
    assert_eq!(*history.try_remove(3, &1).unwrap().unwrap(), 10);
    assert_eq!(*history.try_insert(3, 2, 23).unwrap().unwrap(), 20);
    history.commit(3).unwrap();
    assert!(history.try_insert(5, 1, 50).unwrap().is_none());
    assert_eq!(*history.try_insert(5, 2, 25).unwrap().unwrap(), 23);
    history.commit(5).unwrap();
    assert_eq!(*history.try_remove(7, &2).unwrap().unwrap(), 25);
    history.commit(7).unwrap();
    history.commit(5).unwrap();
    assert!(history.try_insert(9, 3, 90).unwrap().is_none());

    for (txn_id, expected) in [
        (0, vec![(1, 10), (2, 20)]),
        (2, vec![(1, 10), (2, 20)]),
        (3, vec![(2, 23)]),
        (4, vec![(2, 23)]),
        (5, vec![(1, 50), (2, 25)]),
        (6, vec![(1, 50), (2, 25)]),
        (7, vec![(1, 50)]),
        (8, vec![(1, 50)]),
        (9, vec![(1, 50), (3, 90)]),
    ] {
        let snapshot = history
            .iter(txn_id)
            .await
            .unwrap()
            .map(|(key, value)| (*key, *value))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(snapshot, expected.into_iter().collect());
        assert!(history.get(txn_id, &99).await.unwrap().is_none());
    }

    history.finalize(4).unwrap();
    assert_eq!(history.get(3, &2).await.unwrap_err(), Error::Outdated);
    assert_eq!(*history.get(4, &2).await.unwrap().unwrap(), 23);
    assert_eq!(*history.get(5, &1).await.unwrap().unwrap(), 50);
    assert!(history.get(7, &2).await.unwrap().is_none());
    assert_eq!(*history.get(9, &3).await.unwrap().unwrap(), 90);
    assert_eq!(history.rollback(&5), Err(Error::Committed));
    history.finalize(8).unwrap();
    assert_eq!(*history.get(8, &1).await.unwrap().unwrap(), 50);
    assert_eq!(*history.get(9, &3).await.unwrap().unwrap(), 90);
    history.finalize(10).unwrap();
    history.finalize(10).unwrap();
    assert!(history.get(10, &3).await.unwrap().is_none());
    assert_eq!(history.commit(9), Err(Error::Outdated));
    assert_eq!(
        history
            .iter(11)
            .await
            .unwrap()
            .map(|(key, value)| (*key, *value))
            .collect::<BTreeMap<_, _>>(),
        BTreeMap::from([(1, 50)])
    );
}

#[tokio::test]
async fn repeated_read_decisions_return_changes_only_once() {
    let map = TxnMapLock::with_contents(1, [(1, 10)]);
    let (first, changes) = map.read_and_commit(1).await.unwrap();
    assert_eq!(changes.unwrap().len(), 1);
    let (repeated, changes) = map.read_and_commit(1).await.unwrap();
    assert_eq!(repeated, first);
    assert!(changes.is_none());

    map.insert(2, 1, 20).await.unwrap();
    let (second, changes) = map.read_and_commit(2).await.unwrap();
    assert_eq!(changes.unwrap().len(), 1);
    let (repeated, changes) = map.read_and_commit(1).await.unwrap();
    assert_eq!(repeated, first);
    assert!(changes.is_none());
    map.finalize(2).unwrap();
    let (repeated, changes) = map.read_and_commit(2).await.unwrap();
    assert_eq!(repeated, second);
    assert!(changes.is_none());

    let set = TxnSetLock::with_contents(1, [10]);
    let (first, changes) = set.read_and_commit(1).await.unwrap();
    assert_eq!(changes.unwrap().len(), 1);
    let (repeated, changes) = set.read_and_commit(1).await.unwrap();
    assert_eq!(repeated, first);
    assert!(changes.is_none());

    set.insert(2, 20).await.unwrap();
    let (second, changes) = set.read_and_commit(2).await.unwrap();
    assert_eq!(changes.unwrap().len(), 1);
    let (repeated, changes) = set.read_and_commit(1).await.unwrap();
    assert_eq!(repeated, first);
    assert!(changes.is_none());
    set.finalize(2).unwrap();
    let (repeated, changes) = set.read_and_commit(2).await.unwrap();
    assert_eq!(repeated, second);
    assert!(changes.is_none());
}

#[tokio::test]
async fn entries_return_their_keys_and_set_decisions_are_fallible() {
    let map = TxnMapLock::new(1);
    let entry = map.entry(1, 9).await.unwrap();
    assert_eq!(*entry.key(), 9);
    let Entry::Vacant(entry) = entry else {
        panic!("vacant entry")
    };

    drop(entry.insert(90).unwrap());
    let entry = map.entry(1, 9).await.unwrap();
    assert_eq!(*entry.key(), 9);
    assert!(matches!(entry, Entry::Occupied(_)));
    drop(entry);
    drop(map.remove(1, &9).await.unwrap());
    let Entry::Vacant(entry) = map.entry(1, 9).await.unwrap() else {
        panic!("removed entry is vacant")
    };

    assert!(entry.get_committed().unwrap().is_none());
    drop(entry.insert(91).unwrap());
    map.commit(1).unwrap();
    drop(map.remove(2, &9).await.unwrap());
    let Entry::Vacant(entry) = map.entry(2, 9).await.unwrap() else {
        panic!("removed committed entry is vacant")
    };

    assert_eq!(**entry.get_committed().unwrap().as_ref().unwrap(), 91);
    drop(entry.insert(92).unwrap());
    let discarded = map.read_and_rollback(2).await.unwrap();
    assert_eq!(*map.get(2, &9).await.unwrap().unwrap(), 91);
    assert_eq!(**discarded.get(&9).unwrap(), 92);

    let set = TxnSetLock::with_contents(1, [1, 2]);
    set.commit(1).unwrap();
    set.commit(1).unwrap();
    assert_eq!(set.rollback(&1), Err(Error::Committed));
    assert!(set.contains(2, &1).await.unwrap());
    set.finalize(2).unwrap();
    assert_eq!(set.commit(1), Err(Error::Outdated));
    assert_eq!(set.read_and_commit(1).await.unwrap_err(), Error::Outdated);
    assert_eq!(set.rollback(&1), Err(Error::Outdated));

    let history = TxnSetLock::with_contents(1, [1, 2]);
    history.commit(1).unwrap();
    assert!(history.try_remove(3, &1).unwrap());
    history.commit(3).unwrap();
    history.try_insert(5, 1).unwrap();
    history.try_insert(5, 3).unwrap();
    history.commit(5).unwrap();
    assert!(history.try_remove(7, &2).unwrap());
    history.commit(7).unwrap();
    history.commit(5).unwrap();
    history.try_insert(9, 4).unwrap();

    for (txn_id, expected) in [
        (1, vec![1, 2]),
        (2, vec![1, 2]),
        (3, vec![2]),
        (4, vec![2]),
        (5, vec![1, 2, 3]),
        (6, vec![1, 2, 3]),
        (7, vec![1, 3]),
        (8, vec![1, 3]),
        (9, vec![1, 3, 4]),
    ] {
        let snapshot = history
            .iter(txn_id)
            .await
            .unwrap()
            .map(|key| *key)
            .collect::<BTreeSet<_>>();

        assert_eq!(snapshot, expected.into_iter().collect());
        assert!(!history.contains(txn_id, &99).await.unwrap());
    }

    history.finalize(4).unwrap();
    assert_eq!(history.contains(3, &2).await.unwrap_err(), Error::Outdated);
    assert!(history.contains(4, &2).await.unwrap());
    assert!(history.contains(5, &1).await.unwrap());
    assert!(!history.contains(7, &2).await.unwrap());
    assert!(history.contains(9, &4).await.unwrap());
    assert_eq!(history.rollback(&5), Err(Error::Committed));
    history.finalize(8).unwrap();
    assert!(history.contains(8, &1).await.unwrap());
    assert!(history.contains(9, &4).await.unwrap());
    history.finalize(10).unwrap();
    history.finalize(10).unwrap();
    assert!(!history.contains(10, &4).await.unwrap());
    assert_eq!(history.commit(9), Err(Error::Outdated));
    assert_eq!(
        history
            .iter(11)
            .await
            .unwrap()
            .map(|key| *key)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 3])
    );
}

#[test]
fn message_queue_rejections_preserve_other_transactions() {
    let queue = MessageQueue::new(2);
    queue.push(1, 10).unwrap();
    queue.push(2, 20).unwrap();
    assert_eq!(queue.commit(1).unwrap(), vec![10]);
    assert_eq!(queue.rollback(&1), Err(Error::Committed));
    assert!(queue.commit(1).unwrap().is_empty());
    queue.finalize(1).unwrap();
    assert_eq!(queue.commit(1), Err(Error::Outdated));
    assert_eq!(queue.rollback(&1), Err(Error::Outdated));
    assert_eq!(queue.commit(2).unwrap(), vec![20]);
    assert!(queue.commit(2).unwrap().is_empty());
    queue.push(3, 30).unwrap();
    queue.push(4, 40).unwrap();
    assert_eq!(queue.commit(4).unwrap(), vec![40]);
    queue.finalize(3).unwrap();
    assert_eq!(queue.commit(3), Err(Error::Outdated));
    assert_eq!(queue.rollback(&4), Err(Error::Committed));
    assert!(queue.commit(4).unwrap().is_empty());
    queue.push(5, 50).unwrap();
    queue.finalize(4).unwrap();
    queue.finalize(4).unwrap();
    assert_eq!(queue.commit(4), Err(Error::Outdated));
    assert_eq!(queue.commit(5).unwrap(), vec![50]);
    assert!(queue.commit(5).unwrap().is_empty());
}

#[test]
fn poisoned_scalar_state_returns_interrupted_without_reuse() {
    #[derive(Debug)]
    struct PanickingClone;

    impl Clone for PanickingClone {
        fn clone(&self) -> Self {
            panic!("injected caller clone failure")
        }
    }

    let lock = TxnLock::new(PanickingClone);
    assert!(catch_unwind(AssertUnwindSafe(|| lock.try_write(1))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| lock.try_read(1))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| lock.commit(1))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| lock.rollback(&1))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| lock.finalize(1))).is_err());
}

#[tokio::test]
async fn rejected_read_decisions_do_not_register_reservations() {
    let scalar = TxnLock::new(0);
    scalar.commit(1).unwrap();
    assert_eq!(
        scalar.read_and_rollback(1).await.unwrap_err(),
        Error::Committed
    );

    drop(scalar.try_write(2).unwrap());
    scalar.finalize(3).unwrap();
    assert_eq!(
        scalar.read_and_commit(1).await.unwrap_err(),
        Error::Outdated
    );

    assert_eq!(
        scalar.read_and_rollback(1).await.unwrap_err(),
        Error::Outdated
    );

    drop(scalar.try_write(4).unwrap());

    let map = TxnMapLock::with_contents(1, [(1, 1)]);
    map.commit(1).unwrap();
    assert_eq!(
        map.read_and_rollback(1).await.unwrap_err(),
        Error::Committed
    );

    map.try_insert(2, 1, 2).unwrap();
    map.finalize(3).unwrap();
    assert_eq!(map.read_and_commit(1).await.unwrap_err(), Error::Outdated);
    assert_eq!(map.read_and_rollback(1).await.unwrap_err(), Error::Outdated);
    map.try_insert(4, 1, 4).unwrap();

    let set = TxnSetLock::with_contents(1, [1]);
    set.commit(1).unwrap();
    assert_eq!(
        set.read_and_rollback(1).await.unwrap_err(),
        Error::Committed
    );

    set.try_insert(2, 1).unwrap();
    set.finalize(3).unwrap();
    assert_eq!(set.read_and_commit(1).await.unwrap_err(), Error::Outdated);
    assert_eq!(set.read_and_rollback(1).await.unwrap_err(), Error::Outdated);
    set.try_insert(4, 1).unwrap();
}

#[tokio::test]
async fn finalized_frontier_rejects_new_mutations_but_remains_readable() {
    tokio::time::timeout(std::time::Duration::from_millis(200), async {
        let scalar = TxnLock::new(7);
        scalar.finalize(4).unwrap();
        assert_eq!(scalar.try_write(4).unwrap_err(), Error::Outdated);
        assert_eq!(*scalar.read(4).await.unwrap(), 7);
        assert_eq!(*scalar.read_and_commit(4).await.unwrap(), 7);
        assert_eq!(*scalar.read_and_rollback(4).await.unwrap(), 7);
        scalar.rollback(&4).unwrap();
        *scalar.write(5).await.unwrap() = 8;

        let map = TxnMapLock::with_contents(1, [(1, 7)]);
        map.commit(1).unwrap();
        map.finalize(4).unwrap();
        assert_eq!(map.insert(4, 2, 8).await.unwrap_err(), Error::Outdated);
        assert_eq!(*map.get(4, &1).await.unwrap().unwrap(), 7);
        map.insert(5, 2, 8).await.unwrap();
        map.commit(4).unwrap();
        assert_eq!(map.commit(3), Err(Error::Outdated));
        assert_eq!(*map.get(5, &2).await.unwrap().unwrap(), 8);
        map.commit(5).unwrap();
        map.finalize(5).unwrap();
        map.commit(5).unwrap();
        assert_eq!(map.commit(4), Err(Error::Outdated));

        let set = TxnSetLock::with_contents(1, [7]);
        set.commit(1).unwrap();
        set.finalize(4).unwrap();
        assert_eq!(set.insert(4, 8).await.unwrap_err(), Error::Outdated);
        assert!(set.contains(4, &7).await.unwrap());
        assert!(matches!(set.iter(3).await, Err(Error::Outdated)));
        assert!(matches!(set.try_iter(3), Err(Error::Outdated)));
        assert_eq!(set.len(3).await.unwrap_err(), Error::Outdated);
        assert_eq!(set.is_empty(3).await.unwrap_err(), Error::Outdated);
        set.insert(5, 8).await.unwrap();
        set.commit(4).unwrap();
        assert_eq!(set.commit(3), Err(Error::Outdated));
        assert!(set.contains(5, &8).await.unwrap());
        for (txn_id, expected) in [(4, BTreeSet::from([7])), (5, BTreeSet::from([7, 8]))] {
            assert_eq!(
                set.iter(txn_id)
                    .await
                    .unwrap()
                    .map(|key| *key)
                    .collect::<BTreeSet<_>>(),
                expected
            );

            assert_eq!(
                set.try_iter(txn_id)
                    .unwrap()
                    .map(|key| *key)
                    .collect::<BTreeSet<_>>(),
                expected
            );

            assert_eq!(set.len(txn_id).await.unwrap(), expected.len());
            assert!(!set.is_empty(txn_id).await.unwrap());
        }

        set.commit(5).unwrap();
        set.finalize(5).unwrap();
        set.commit(5).unwrap();
        assert_eq!(set.commit(4), Err(Error::Outdated));

        for (txn_id, cutoff, expected_error, next) in [
            (1, None, Error::Committed, 2),
            (3, Some(4), Error::Outdated, 5),
            (4, Some(4), Error::Outdated, 5),
        ] {
            for operation in 0..3 {
                let set = TxnSetLock::with_contents(1, [7]);
                set.commit(1).unwrap();
                if let Some(cutoff) = cutoff {
                    set.finalize(cutoff).unwrap();
                }

                let result = match operation {
                    0 => set.try_clear(txn_id).map(|_| ()),
                    1 => set.try_extend(txn_id, [8]),
                    _ => set.try_insert(txn_id, 8),
                };

                assert_eq!(result, Err(expected_error.clone()));
                assert_eq!(set.len(next).await.unwrap(), 1);
                set.insert(next, 8).await.unwrap();
                assert_eq!(set.len(next).await.unwrap(), 2);
            }
        }
    })
    .await
    .expect("rejected cutoff work must not register a blocking reservation");
}

#[tokio::test]
async fn set_removal_waits_for_held_transactional_read_access() {
    let set = TxnSetLock::with_contents(1, [7]);
    set.commit(1).unwrap();
    let values = set.iter(2).await.unwrap();
    let mut removal = std::pin::pin!(set.remove(2, &7));
    assert!(futures::poll!(&mut removal).is_pending());
    drop(values);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(200), removal)
            .await
            .unwrap()
            .unwrap()
    );
}

#[tokio::test]
async fn yielded_map_guards_retain_access_after_iterator_drop() {
    let map = TxnMapLock::with_contents(1, [(1, 7)]);
    for mutable in [false, true] {
        if mutable {
            let mut iter = map.iter_mut(1).await.unwrap();
            let (_, mut value) = iter.next().unwrap();
            drop(iter);
            assert_eq!(map.try_get_mut(1, &1).unwrap_err(), Error::WouldBlock);
            assert_eq!(map.commit(1), Err(Error::Busy));
            *value = 8;
            drop(value);
        } else {
            let mut iter = map.iter(1).await.unwrap();
            let (_, value) = iter.next().unwrap();
            drop(iter);
            assert_eq!(*value, 7);
            assert_eq!(map.try_get_mut(1, &1).unwrap_err(), Error::WouldBlock);
            assert_eq!(map.commit(1), Err(Error::Busy));
            drop(value);
        }

        assert!(map.try_get_mut(1, &1).unwrap().is_some());
    }

    map.commit(1).unwrap();
    assert_eq!(*map.get(2, &1).await.unwrap().unwrap(), 8);
}

#[tokio::test]
async fn vacant_owner_inspection_rejects_newer_exact_key_changes() {
    for deleted in [false, true] {
        let map = TxnMapLock::from_committed(if deleted { vec![(9, 90)] } else { vec![] });
        assert_eq!(
            map.get(2, &9).await.unwrap().map(|value| *value),
            deleted.then_some(90)
        );

        if deleted {
            drop(map.remove(3, &9).await.unwrap());
        } else {
            map.insert(3, 9, 93).await.unwrap();
        }

        map.commit(3).unwrap();
        // A newer change to another name does not make this owner obsolete.
        let Entry::Vacant(other) = map.entry(2, 8).await.unwrap() else {
            panic!("vacant name")
        };

        assert!(other.get_committed().unwrap().is_none());
        drop(other);
        if deleted {
            drop(map.remove(2, &9).await.unwrap());
        }

        let Entry::Vacant(entry) = map.entry(2, 9).await.unwrap() else {
            panic!("historically vacant name")
        };

        assert_eq!(entry.get_committed().unwrap_err(), Error::Conflict);
        assert_eq!(map.try_insert(4, 9, 94).unwrap_err(), Error::WouldBlock);
        drop(entry);
        // Dropping active access does not erase this transaction's reservations.
        assert_eq!(map.try_insert(4, 9, 94).unwrap_err(), Error::WouldBlock);
        map.rollback(&2).unwrap();
        assert_eq!(
            map.get(2, &9).await.unwrap().map(|value| *value),
            deleted.then_some(90)
        );

        assert_eq!(
            map.get(4, &9).await.unwrap().map(|value| *value),
            (!deleted).then_some(93)
        );

        map.insert(4, 9, 94).await.unwrap();
        map.commit(4).unwrap();
        map.finalize(3).unwrap();
        assert_eq!(map.entry(2, 9).await.err().unwrap(), Error::Outdated);
        drop(map.remove(5, &9).await.unwrap());
        let Entry::Vacant(entry) = map.entry(5, 9).await.unwrap() else {
            panic!("removed owner")
        };

        assert_eq!(**entry.get_committed().unwrap().as_ref().unwrap(), 94);
        drop(entry.insert(95).unwrap());
        map.commit(5).unwrap();
        assert_eq!(*map.get(5, &9).await.unwrap().unwrap(), 95);
    }
}
