import pytest

from data_utils import ACTIVE_TASKS, PASSIVE_TASKS, parallel_filter, resolve_tasks


def test_resolve_tasks_passive_shorthand():
    assert resolve_tasks("passive") == PASSIVE_TASKS


def test_resolve_tasks_active_shorthand():
    assert resolve_tasks("active") == ACTIVE_TASKS


def test_resolve_tasks_all_shorthand():
    assert resolve_tasks("all") == PASSIVE_TASKS + ACTIVE_TASKS


def test_resolve_tasks_is_case_insensitive():
    assert resolve_tasks("PASSIVE") == PASSIVE_TASKS


def test_resolve_tasks_single_valid_task_name_stays_a_string():
    assert resolve_tasks("contrastChangeDetection") == "contrastChangeDetection"


def test_resolve_tasks_valid_list_is_returned_unchanged():
    tasks = ["contrastChangeDetection", "RestingState"]
    assert resolve_tasks(tasks) == tasks


def test_resolve_tasks_invalid_string_raises_assertion_error():
    with pytest.raises(AssertionError):
        resolve_tasks("not_a_real_task")


def test_resolve_tasks_invalid_list_raises_assertion_error():
    with pytest.raises(AssertionError):
        resolve_tasks(["not_a_real_task"])


def test_resolve_tasks_list_with_lowercase_keyword_is_not_expanded():
    # Only bare strings get the passive/active/all shorthand treatment.
    with pytest.raises(AssertionError):
        resolve_tasks(["passive"])


def test_resolve_tasks_invalid_type_raises_value_error():
    """Regression test: the original code built a ValueError but never raised it."""
    with pytest.raises(ValueError):
        resolve_tasks(123)


def test_parallel_filter_keeps_only_matching_items():
    def is_even(x: int) -> bool:
        return x % 2 == 0

    result = parallel_filter(is_even, list(range(10)), n_jobs=1)

    assert result == [0, 2, 4, 6, 8]


def test_parallel_filter_preserves_order():
    def not_three(x: int) -> bool:
        return x != 3

    result = parallel_filter(not_three, [3, 1, 3, 2, 3], n_jobs=1)

    assert result == [1, 2]
