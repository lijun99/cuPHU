"""Tests for cuphu.build_water_mask()."""

import numpy as np
import pytest

import cuphu


def test_differentiated_matches_isce3_formula():
    """Reference formula from isce3's _build_invalid_mask()."""
    rng = np.random.default_rng(0)
    water_distance = rng.integers(0, 201, size=(50, 50)).astype(np.float32)
    ocean_buf, inland_buf = 3.0, 2.0

    inland_ref = water_distance > (inland_buf + 100)
    ocean_ref = (water_distance > ocean_buf) & (water_distance <= 100)
    expected = inland_ref | ocean_ref

    result = cuphu.build_water_mask(
        water_distance, ocean_water_buffer=ocean_buf, inland_water_buffer=inland_buf
    )
    np.testing.assert_array_equal(result, expected)


def test_differentiated_land_never_invalid():
    water_distance = np.zeros((10, 10), dtype=np.float32)
    result = cuphu.build_water_mask(
        water_distance, ocean_water_buffer=0.0, inland_water_buffer=0.0
    )
    assert not result.any()


def test_uniform_grows_valid_region_by_exact_pixels():
    water_distance = np.zeros((21, 21), dtype=np.float32)
    water_distance[10, 10] = 0  # single land pixel
    water_distance[:] = 1  # everything else water
    water_distance[10, 10] = 0

    result = cuphu.build_water_mask(water_distance, water_buffer=3)
    valid = ~result
    ys, xs = np.where(valid)
    manhattan = np.abs(ys - 10) + np.abs(xs - 10)
    assert manhattan.max() == 3


def test_uniform_zero_buffer_is_plain_indicator():
    water_distance = np.array([[0, 1], [2, 0]], dtype=np.float32)
    result = cuphu.build_water_mask(water_distance, water_buffer=0)
    np.testing.assert_array_equal(result, water_distance != 0)


def test_rejects_both_modes_together():
    water_distance = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        cuphu.build_water_mask(
            water_distance,
            ocean_water_buffer=1.0,
            inland_water_buffer=1.0,
            water_buffer=1,
        )


def test_rejects_partial_differentiated_args():
    water_distance = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        cuphu.build_water_mask(water_distance, ocean_water_buffer=1.0)
    with pytest.raises(ValueError):
        cuphu.build_water_mask(water_distance, inland_water_buffer=1.0)


def test_rejects_no_args():
    water_distance = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        cuphu.build_water_mask(water_distance)
