"""The error-injection script keeps batches at 10 rows and its errors are
detectable inside a single batch."""

import numpy as np
import pandas as pd
import pytest

from scripts.inject_errors import ROW_ERRORS, inject

NO_ERRORS = {e: 0.0 for e in ROW_ERRORS} | {"missing_column": 0.0}


def _batch():
    return pd.DataFrame(
        {
            "Season": [2023] * 10,
            "Round": list(range(1, 11)),
            "Grid": list(range(1, 11)),
            "CircuitID": ["monza"] * 10,
            "DriverID": [f"driver_{i}" for i in range(10)],
            "ConstructorName": ["Ferrari"] * 10,
        }
    )


@pytest.mark.parametrize("error", ROW_ERRORS)
def test_row_count_never_changes(error):
    out, _ = inject(_batch(), NO_ERRORS | {error: 1.0}, np.random.default_rng(0))
    assert len(out) == 10


def test_zero_probability_leaves_data_untouched():
    out, counts = inject(_batch(), NO_ERRORS, np.random.default_rng(0))
    pd.testing.assert_frame_equal(out.astype(str), _batch().astype(str))
    assert all(counts[e] == 0 for e in ROW_ERRORS)


def test_duplicates_land_inside_the_batch():
    probs = NO_ERRORS | {"duplicate_rows": 0.3}
    out, counts = inject(_batch(), probs, np.random.default_rng(1))
    assert counts["duplicate_rows"] > 0
    assert out.duplicated(keep=False).any()


def test_out_of_range_values_are_real_violations():
    out, _ = inject(
        _batch(), NO_ERRORS | {"out_of_range": 1.0}, np.random.default_rng(0)
    )
    grid = out["Grid"].astype(int)
    assert ((grid < 0) | (grid > 40)).all()


def test_missing_column_drops_one_required_column():
    out, counts = inject(
        _batch(), NO_ERRORS | {"missing_column": 1.0}, np.random.default_rng(0)
    )
    assert len(out.columns) == 5
    assert counts["missing_column"].startswith("dropped")
