"""The export ETA, including the case that once showed a negative time."""
import pytest

from diareader.gui.main_window import eta_suffix


def test_no_estimate_before_there_is_anything_to_average():
    assert eta_suffix(done=0, total=100, elapsed=0.0) == ""
    assert eta_suffix(done=1, total=100, elapsed=20.0) == ""


def test_seconds_then_minutes_then_hours():
    # 2 done in 20s -> 10s each.
    assert eta_suffix(done=2, total=6, elapsed=20.0) == " — about 40s left"
    assert eta_suffix(done=2, total=30, elapsed=20.0) == " — about 5 min left"
    assert eta_suffix(done=2, total=1000, elapsed=20.0) == " — about 2.8 h left"


def test_a_total_below_the_work_already_done_never_reads_as_negative():
    """What a real run showed. `total` was shared with the chapter-list selection,
    and the list stays usable during an export — so selecting a shorter range
    mid-run pushed the total under chunks_done and the status line offered a
    negative number of minutes remaining."""
    suffix = eta_suffix(done=50, total=10, elapsed=500.0)
    assert "-" not in suffix, suffix
    assert suffix == " — about 0s left"


@pytest.mark.parametrize("total", [0, -5])
def test_no_estimate_when_the_total_is_meaningless(total):
    assert eta_suffix(done=10, total=total, elapsed=100.0) == ""


def test_finishing_exactly_on_the_total_reads_as_done():
    assert eta_suffix(done=10, total=10, elapsed=100.0) == " — about 0s left"
