"""Model simulation and verification: numerical ODE solutions vs analytic laws."""
import numpy as np

from src import braking as B
from src import clausius_clapeyron as CC


# ---------- braking ----------
def test_rk4_braking_distance_matches_analytic():
    assert abs(B.rk4(0.01)[1][-1] - B.analytic_braking_distance(20.0)) < 1e-9


def test_euler_braking_distance_close_to_analytic():
    assert abs(B.euler(0.01)[1][-1] - B.analytic_braking_distance(20.0)) < 0.2


def test_total_stopping_distance_is_43p125():
    assert abs(B.analytic_stopping_distance(20.0) - 43.1248) < 1e-4


def test_euler_braking_is_first_order():
    dts = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    err = [abs(B.analytic_braking_distance(20.0) - B.euler(d)[1][-1]) for d in dts]
    assert abs(np.polyfit(np.log(dts), np.log(err), 1)[0] - 1.0) < 0.05


def test_energy_conserved_by_rk4_and_drifts_for_euler():
    _, xk, vk = B.rk4(0.01)
    _, xe, ve = B.euler(0.01)
    E0 = 0.5 * 1200 * 20.0**2
    assert np.abs(B.energy(xk, vk) - E0).max() / E0 < 1e-12
    assert 1e-3 < np.abs(B.energy(xe, ve) - E0).max() / E0 < 1e-2


# ---------- Clausius-Clapeyron ----------
T0, P0, TEND, L = 293.15, 2.329575345, 373.15, 43.212e3


def test_rk4_matches_analytic_law():
    T, P = CC.rk4(T0, P0, TEND, 40, L)
    assert np.max(np.abs(P - CC.analytic(T, P0, T0, L)) / CC.analytic(T, P0, T0, L)) < 1e-5


def test_euler_error_is_visible_at_coarse_step():
    T, P = CC.euler(T0, P0, TEND, 40, L)
    assert abs(P[-1] - CC.analytic(T[-1], P0, T0, L)) / CC.analytic(T[-1], P0, T0, L) > 0.1


def test_measured_orders_are_1_and_4():
    Ns = 2 ** np.arange(1, 15)
    hs = (TEND - T0) / Ns
    e_euler, e_rk4 = [], []
    for n in Ns:
        for solver, store in ((CC.euler, e_euler), (CC.rk4, e_rk4)):
            T, P = solver(T0, P0, TEND, int(n), L)
            ex = CC.analytic(T[-1], P0, T0, L)
            store.append(abs(P[-1] - ex) / ex)
    assert abs(CC.measured_order(hs, e_euler) - 1) < 0.05
    assert abs(CC.measured_order(hs, e_rk4) - 4) < 0.15


def test_invariant_conserved_within_stated_tolerances():
    I0 = np.log(P0) + L / (CC.R * T0)
    T, P = CC.rk4(T0, P0, TEND, 1000, L)
    assert np.abs(CC.invariant(T, P, L) - I0).max() < 1e-10
    T, P = CC.euler(T0, P0, TEND, 1000, L)
    assert np.abs(CC.invariant(T, P, L) - I0).max() < 1e-2


def test_limit_zero_latent_heat_keeps_pressure_constant():
    _, P = CC.rk4(T0, P0, TEND, 200, 0.0)
    assert np.allclose(P, P0, rtol=0, atol=1e-14)


def test_limit_large_temperature_saturates():
    assert np.isclose(CC.analytic(1e9, P0, T0, L), P0 * np.exp(L / (CC.R * T0)), rtol=1e-5)
