"""Data and statistics: fitted parameters reproduce the reported results."""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

R = 8.314462618


def test_stopping_fit_reaction_time_consistent_with_0p7(stopping):
    v, d, sd = stopping.v_ms.values, stopping.d_sim.values, stopping.sigma.values
    model = lambda v, tr, mu: v * tr + v**2 / (2 * mu * 9.81)
    popt, pcov = curve_fit(model, v, d, p0=[1.0, 0.5], sigma=sd, absolute_sigma=True)
    tr, mu = popt
    tr_err, mu_err = np.sqrt(np.diag(pcov))
    assert abs(tr - 0.689) < 0.002 and abs(tr_err - 0.026) < 0.002
    assert abs(mu - 0.682) < 0.002 and abs(mu_err - 0.015) < 0.002
    t = (tr - 0.7) / tr_err
    p = 2 * stats.t.sf(abs(t), len(v) - 2)
    assert abs(t + 0.43) < 0.02 and abs(p - 0.669) < 0.005 and p > 0.05
    assert np.sqrt(np.mean((d - model(v, tr, mu)) ** 2)) < 3.2


def test_vapour_weighted_fit_latent_heat(vapour):
    tr = vapour[vapour.split == "train"]
    T0 = float(vapour.T_K.iloc[0])
    x = 1 / tr.T_K.to_numpy() - 1 / T0
    y = np.log(tr.P_sim.to_numpy())
    sig = tr.sigma.to_numpy() / tr.P_sim.to_numpy()
    w = 1 / sig**2
    X = np.column_stack([np.ones(len(x)), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    ln_p0, slope = cov @ (X.T @ (w * y))
    L, L_se = -slope * R / 1000, R * np.sqrt(cov[1, 1]) / 1000
    assert abs(L - 43.128) < 0.01 and abs(L_se - 0.295) < 0.005
    assert abs(np.exp(ln_p0) - 2.3726) < 0.001
    chi2_red = np.sum(((y - X @ np.array([ln_p0, slope])) / sig) ** 2) / (len(x) - 2)
    assert abs(chi2_red - 1.370) < 0.01
