import pytest

pytest.importorskip("datasets")

from lerobot.datasets.factory import split_episodes_per_task


def test_last_mode_holds_out_tail_per_task():
    tasks = {"a": list(range(10)), "b": list(range(10, 15))}
    train, ev = split_episodes_per_task(tasks, 0.2, "last", seed=0)
    assert ev == [8, 9, 14]
    assert sorted(train + ev) == list(range(15))


def test_random_mode_is_seeded_and_stratified():
    tasks = {"a": list(range(10)), "b": list(range(10, 20))}
    t1, e1 = split_episodes_per_task(tasks, 0.2, "random", seed=3)
    t2, e2 = split_episodes_per_task(tasks, 0.2, "random", seed=3)
    assert (t1, e1) == (t2, e2)
    assert len([e for e in e1 if e < 10]) == 2 and len([e for e in e1 if e >= 10]) == 2
    assert set(t1).isdisjoint(e1)
    assert split_episodes_per_task(tasks, 0.2, "random", seed=4)[1] != e1


def test_single_episode_task_stays_in_train():
    train, ev = split_episodes_per_task({"a": [0], "b": [1, 2, 3, 4, 5]}, 0.2, "last", seed=0)
    assert 0 in train and ev == [5]
    train, ev = split_episodes_per_task({"a": [0]}, 0.5, "random", seed=0)
    assert train == [0] and ev == []
