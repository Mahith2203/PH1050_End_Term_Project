"""Data tests 2: integrity and physics checks on the vapour-pressure dataset."""
import numpy as np

R = 8.314462618


def test_temperature_runs_20_to_100_c(vapour):
    assert abs(vapour.T_C.min() - 20) < 1e-5 and abs(vapour.T_C.max() - 100) < 1e-5


def test_temperature_steps_are_equal(vapour):
    assert np.diff(vapour.T_C).max() - np.diff(vapour.T_C).min() < 1e-5


def test_kelvin_equals_celsius_plus_273p15(vapour):
    assert np.abs(vapour.T_K - (vapour.T_C + 273.15)).max() < 1e-5


def test_all_values_positive(vapour):
    assert (vapour[["T_K", "P_ref", "P_sim", "sigma"]] > 0).all().all()


def test_uncertainty_is_3_percent_of_simulated(vapour):
    assert np.abs(vapour.sigma - 0.03 * vapour.P_sim).max() < 1e-5


def test_reference_pressure_increases_with_temperature(vapour):
    assert (np.diff(vapour.P_ref) > 0).all()


def test_ln_p_vs_inverse_t_is_straight(vapour):
    assert np.corrcoef(1 / vapour.T_K, np.log(vapour.P_sim))[0, 1] < -0.99


def test_noise_is_about_3_percent(vapour):
    noise = vapour.P_sim / vapour.P_ref - 1
    assert 0.015 < np.sqrt(np.mean(noise**2)) < 0.06


def test_no_outliers_within_15_percent(vapour):
    assert np.abs(vapour.P_sim / vapour.P_ref - 1).max() < 0.15


def test_pressure_at_100c_is_one_atmosphere(vapour):
    assert abs(vapour.P_ref.iloc[-1] - 101.325) / 101.325 < 0.01


def test_latent_heat_between_38_and_46_kj_per_mol(vapour):
    slope, _ = np.polyfit(1 / vapour.T_K, np.log(vapour.P_ref), 1)
    assert 38 < -slope * R / 1000 < 46


def test_labels_are_train_and_test(vapour):
    assert set(vapour.split) == {"train", "test"}


def test_36_train_and_24_test_rows(vapour):
    assert (vapour.split == "train").sum() == 36 and (vapour.split == "test").sum() == 24


def test_train_rows_precede_test_rows(vapour):
    assert vapour.T_K[vapour.split == "train"].max() < vapour.T_K[vapour.split == "test"].min()
