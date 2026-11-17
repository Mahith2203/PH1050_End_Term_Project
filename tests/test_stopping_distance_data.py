"""Data tests 1: integrity and physics checks on the stopping-distance dataset."""
import numpy as np

G = 9.81


def test_speed_range_20_to_120_kmh(stopping):
    assert abs(stopping.v_kmh.min() - 20) < 1e-5 and abs(stopping.v_kmh.max() - 120) < 1e-5


def test_speed_steps_are_equal(stopping):
    assert np.diff(stopping.v_kmh).max() - np.diff(stopping.v_kmh).min() < 1e-5


def test_ms_equals_kmh_over_3p6(stopping):
    assert np.abs(stopping.v_ms - stopping.v_kmh / 3.6).max() < 1e-5


def test_all_values_positive(stopping):
    assert (stopping[["v_ms", "d_theo", "d_sim", "sigma"]] > 0).all().all()


def test_uncertainty_is_5_percent_of_simulated(stopping):
    assert np.abs(stopping.sigma - 0.05 * stopping.d_sim).max() < 1e-5


def test_theoretical_distance_increases_with_speed(stopping):
    assert (np.diff(stopping.d_theo) > 0).all()


def test_road_grade_factor_runs_1_to_1p02(stopping):
    assert abs(stopping.grade_factor.min() - 1.00) < 1e-5
    assert abs(stopping.grade_factor.max() - 1.02) < 1e-5


def test_noise_is_about_5_percent(stopping):
    noise = stopping.d_sim / (stopping.d_theo * stopping.grade_factor) - 1
    assert 0.025 < np.sqrt(np.mean(noise**2)) < 0.08


def test_no_extreme_outliers(stopping):
    noise = stopping.d_sim / (stopping.d_theo * stopping.grade_factor) - 1
    assert np.abs(noise).max() < 0.25


def test_underlying_reaction_time_is_0p7(stopping):
    slope, intercept = np.polyfit(stopping.v_ms, stopping.d_theo / stopping.v_ms, 1)
    assert abs(intercept - 0.7) < 1e-4


def test_underlying_friction_is_0p7(stopping):
    slope, _ = np.polyfit(stopping.v_ms, stopping.d_theo / stopping.v_ms, 1)
    assert abs(1.0 / (2.0 * G * slope) - 0.7) < 1e-4


def test_labels_are_train_and_test(stopping):
    assert set(stopping.split) == {"train", "test"}


def test_36_train_and_24_test_rows(stopping):
    assert (stopping.split == "train").sum() == 36 and (stopping.split == "test").sum() == 24


def test_train_speeds_precede_test_speeds(stopping):
    assert stopping.v_ms[stopping.split == "train"].max() < stopping.v_ms[stopping.split == "test"].min()


def test_no_missing_values_and_60_rows(stopping):
    assert len(stopping) == 60 and stopping.isna().sum().sum() == 0
