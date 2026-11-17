"""Supplementary statistics and figures for both datasets.

Run from the repository root:  python src/supplementary_analysis.py

Outputs (results/figures and results/tables):
  noise_distributions.png          noise histograms for both datasets
  stopping_parameter_correlation   (t_r, mu) confidence ellipses
  normalised_residuals_qq.png      standardised residuals and Q-Q plots
  vapour_prediction_band.png       fitted law with 95 % band, train/test marked
  stopping_train_only_fit.png      fit on slowest 60 %, evaluated on fastest 40 %
  supplementary_summary.csv        numbers quoted in the README
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
G, R = 9.81, 8.314462618
BLUE, NAVY, ORANGE, GREY = "#1F5FBF", "#0B2A5B", "#E07B00", "#5B6B82"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- data
st = pd.read_csv(ROOT / "data" / "stopping_distance_60_values.csv")
st.columns = ["obs", "v_kmh", "v_ms", "d_theo", "grade", "d_sim", "sigma", "split"]
vp = pd.read_csv(ROOT / "data" / "vapour_pressure_60_values.csv", encoding="latin-1")
vp.columns = ["obs", "T_C", "T_K", "P_ref", "P_sim", "sigma", "split"]
rows = []

# ---------------------------------------------------------------- stopping fit (all 60 points, as in the notebook)
model = lambda v, tr, mu: v * tr + v**2 / (2 * mu * G)
v, d, sd = st.v_ms.values, st.d_sim.values, st.sigma.values
popt, pcov = curve_fit(model, v, d, p0=[1.0, 0.5], sigma=sd, absolute_sigma=True)
tr, mu = popt
tr_err, mu_err = np.sqrt(np.diag(pcov))
corr = pcov[0, 1] / (tr_err * mu_err)
z_st = (d - model(v, tr, mu)) / sd
chi2_st = float(np.sum(z_st**2))
rows += [("stopping_tr_s", tr), ("stopping_tr_err_s", tr_err), ("stopping_mu", mu), ("stopping_mu_err", mu_err),
         ("stopping_corr_tr_mu", corr), ("stopping_chi2_reduced", chi2_st / (len(v) - 2)),
         ("stopping_shapiro_p_z", stats.shapiro(z_st).pvalue)]

# ---------------------------------------------------------------- vapour weighted fit (training rows, as in the report)
T0 = float(vp.T_K.iloc[0])
trn = vp[vp.split == "train"]
x = 1 / trn.T_K.to_numpy() - 1 / T0
y = np.log(trn.P_sim.to_numpy())
sig = trn.sigma.to_numpy() / trn.P_sim.to_numpy()
w = 1 / sig**2
X = np.column_stack([np.ones(len(x)), x])
C = np.linalg.inv(X.T @ (w[:, None] * X))
beta = C @ (X.T @ (w * y))
xa = 1 / vp.T_K.to_numpy() - 1 / T0
fit_ln = beta[0] + beta[1] * xa
z_vp = (np.log(vp.P_sim) - fit_ln) / (vp.sigma / vp.P_sim)
rows += [("vapour_L_kJ_mol", -beta[1] * R / 1000), ("vapour_L_err_kJ_mol", R * np.sqrt(C[1, 1]) / 1000),
         ("vapour_shapiro_p_z_train", stats.shapiro(z_vp[vp.split == "train"]).pvalue),
         ("vapour_z_rms_train", np.sqrt(np.mean(z_vp[vp.split == "train"]**2))),
         ("vapour_z_rms_test", np.sqrt(np.mean(z_vp[vp.split == "test"]**2)))]

# ---------------------------------------------------------------- 1. noise distributions
n_st = st.d_sim / (st.d_theo * st.grade) - 1
n_vp = vp.P_sim / vp.P_ref - 1
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
ax[0, 0].hist(100 * n_st, bins=14, color=BLUE, edgecolor="white")
ax[0, 0].set(title="Stopping distance: relative noise", xlabel="(simulated / truth - 1) in %", ylabel="count")
ax[0, 1].plot(st.v_kmh, 100 * n_st, "o", color=BLUE, ms=4)
ax[0, 1].axhline(0, color="k", lw=1)
ax[0, 1].set(title="Noise against speed", xlabel="Speed (km/h)", ylabel="noise (%)")
ax[1, 0].hist(100 * n_vp, bins=14, color=NAVY, edgecolor="white")
ax[1, 0].set(title="Vapour pressure: relative noise", xlabel="(simulated / reference - 1) in %", ylabel="count")
ax[1, 1].plot(vp.T_C, 100 * n_vp, "o", color=NAVY, ms=4)
ax[1, 1].axhline(0, color="k", lw=1)
ax[1, 1].set(title="Noise against temperature", xlabel="Temperature (deg C)", ylabel="noise (%)")
fig.tight_layout(); fig.savefig(FIG / "noise_distributions.png", dpi=150); plt.close(fig)
rows += [("stopping_noise_rms_pct", 100 * np.sqrt(np.mean(n_st**2))), ("stopping_noise_mean_pct", 100 * n_st.mean()),
         ("vapour_noise_rms_pct", 100 * np.sqrt(np.mean(n_vp**2))), ("vapour_noise_mean_pct", 100 * n_vp.mean())]

# ---------------------------------------------------------------- 2. (t_r, mu) confidence ellipses
fig, ax = plt.subplots(figsize=(5.6, 4.6))
vals, vecs = np.linalg.eigh(pcov)
angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
for q, col, lab in ((stats.chi2.ppf(0.683, 2), BLUE, "68 %"), (stats.chi2.ppf(0.95, 2), NAVY, "95 %")):
    ax.add_patch(Ellipse((tr, mu), 2 * np.sqrt(q * vals[1]), 2 * np.sqrt(q * vals[0]), angle=angle,
                         fill=False, ec=col, lw=2, label=f"{lab} joint region"))
ax.plot(tr, mu, "o", color=BLUE, label="fit")
ax.plot(0.7, 0.7, "*", color=ORANGE, ms=14, label="generating values (0.7, 0.7)")
ax.set(xlabel="reaction time $t_r$ (s)", ylabel="friction coefficient $\\mu$",
       title=f"Joint confidence region (correlation {corr:+.2f})")
ax.autoscale(); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "stopping_parameter_correlation.png", dpi=150); plt.close(fig)
inside = (np.array([0.7 - tr, 0.7 - mu]) @ np.linalg.inv(pcov) @ np.array([0.7 - tr, 0.7 - mu]))
rows += [("stopping_generating_point_mahalanobis_chi2", inside),
         ("stopping_generating_point_p_value", 1 - stats.chi2.cdf(inside, 2))]

# ---------------------------------------------------------------- 3. standardised residuals and Q-Q
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
for r, (zz, xx, xl, col, ttl, mask) in enumerate((
        (z_st, st.v_kmh, "Speed (km/h)", BLUE, "Stopping distance", st.split == "test"),
        (z_vp.values, vp.T_C, "Temperature (deg C)", NAVY, "Vapour pressure", vp.split == "test"))):
    mask = np.asarray(mask)
    ax[r, 0].axhspan(-2, 2, color=GREY, alpha=0.12)
    ax[r, 0].plot(np.asarray(xx)[~mask], np.asarray(zz)[~mask], "o", color=col, ms=4, label="train")
    ax[r, 0].plot(np.asarray(xx)[mask], np.asarray(zz)[mask], "s", color=ORANGE, ms=4, label="test")
    ax[r, 0].axhline(0, color="k", lw=1)
    ax[r, 0].set(title=f"{ttl}: standardised residuals", xlabel=xl, ylabel="(data - fit) / sigma"); ax[r, 0].legend(fontsize=8)
    (osm, osr), (sl, ic, rr) = stats.probplot(np.asarray(zz), dist="norm")
    ax[r, 1].plot(osm, osr, "o", color=col, ms=4)
    ax[r, 1].plot(osm, sl * np.asarray(osm) + ic, "k-", lw=1)
    ax[r, 1].set(title=f"{ttl}: normal Q-Q plot", xlabel="theoretical quantiles", ylabel="ordered values")
fig.tight_layout(); fig.savefig(FIG / "normalised_residuals_qq.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 4. vapour prediction band
Tg = np.linspace(vp.T_K.min(), vp.T_K.max(), 300)
xg = 1 / Tg - 1 / T0
Xg = np.column_stack([np.ones_like(xg), xg])
mean_ln = Xg @ beta
se_ln = np.sqrt(np.einsum("ij,jk,ik->i", Xg, C, Xg))
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(Tg - 273.15, np.exp(mean_ln - 1.96 * se_ln), np.exp(mean_ln + 1.96 * se_ln), color=BLUE, alpha=0.25,
                label="95 % band of the fitted law")
ax.plot(Tg - 273.15, np.exp(mean_ln), color=NAVY, lw=2, label="weighted fit (training rows)")
mt = (vp.split == "train").to_numpy()
ax.errorbar(vp.T_C[mt], vp.P_sim[mt], yerr=vp.sigma[mt], fmt="o", ms=3.5, color=BLUE, label="train")
ax.errorbar(vp.T_C[~mt], vp.P_sim[~mt], yerr=vp.sigma[~mt], fmt="s", ms=3.5, color=ORANGE, label="test (held out)")
ax.set(xlabel="Temperature (deg C)", ylabel="Vapour pressure (kPa)", title="Vapour pressure: fitted law and held-out data")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "vapour_prediction_band.png", dpi=150); plt.close(fig)
rows += [("vapour_test_inside_95pct_band_fraction",
          float(np.mean([(np.log(p) > m - 1.96 * np.sqrt(s2 + (sg)**2)) and (np.log(p) < m + 1.96 * np.sqrt(s2 + (sg)**2))
                         for p, m, s2, sg in zip(vp.P_sim[~mt],
                                                 (np.column_stack([np.ones(24), 1 / vp.T_K[~mt] - 1 / T0]) @ beta),
                                                 np.einsum("ij,jk,ik->i", np.column_stack([np.ones(24), 1 / vp.T_K[~mt] - 1 / T0]), C,
                                                           np.column_stack([np.ones(24), 1 / vp.T_K[~mt] - 1 / T0])),
                                                 (vp.sigma[~mt] / vp.P_sim[~mt]))])))]

# ---------------------------------------------------------------- 5. stopping: fit on slowest 60 %, test on fastest 40 %
mtr = (st.split == "train").to_numpy()
p2, c2 = curve_fit(model, v[mtr], d[mtr], p0=[1.0, 0.5], sigma=sd[mtr], absolute_sigma=True)
pred = model(v, *p2)
def rmse(m): return float(np.sqrt(np.mean((pred[m] - d[m])**2)))
def rrel(m): return float(100 * np.sqrt(np.mean(((pred[m] - d[m]) / d[m])**2)))
e2 = np.sqrt(np.diag(c2))
rows += [("stopping_trainonly_tr_s", p2[0]), ("stopping_trainonly_tr_err_s", e2[0]),
         ("stopping_trainonly_mu", p2[1]), ("stopping_trainonly_mu_err", e2[1]),
         ("stopping_trainonly_rmse_train_m", rmse(mtr)), ("stopping_trainonly_rmse_test_m", rmse(~mtr)),
         ("stopping_trainonly_rms_rel_train_pct", rrel(mtr)), ("stopping_trainonly_rms_rel_test_pct", rrel(~mtr))]
vg = np.linspace(v.min(), v.max(), 300)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].errorbar(v[mtr] * 3.6, d[mtr], yerr=sd[mtr], fmt="o", ms=3.5, color=BLUE, label="train (slowest 60 %)")
ax[0].errorbar(v[~mtr] * 3.6, d[~mtr], yerr=sd[~mtr], fmt="s", ms=3.5, color=ORANGE, label="test (fastest 40 %)")
ax[0].plot(vg * 3.6, model(vg, *p2), color=NAVY, lw=2, label="fit on train only")
ax[0].plot(vg * 3.6, model(vg, 0.7, 0.7), "--", color=GREY, label="generating law ($t_r=\\mu=0.7$)")
ax[0].set(xlabel="Speed (km/h)", ylabel="Stopping distance (m)", title="Fit on the slow data, extrapolate to the fast"); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="k", lw=1)
ax[1].plot(v[mtr] * 3.6, d[mtr] - pred[mtr], "o", color=BLUE, ms=4, label="train")
ax[1].plot(v[~mtr] * 3.6, d[~mtr] - pred[~mtr], "s", color=ORANGE, ms=4, label="test")
ax[1].set(xlabel="Speed (km/h)", ylabel="Residual (m)", title="Residuals of the train-only fit"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "stopping_train_only_fit.png", dpi=150); plt.close(fig)

summary = pd.DataFrame(rows, columns=["quantity", "value"])
summary.to_csv(TAB / "supplementary_summary.csv", index=False)
print(summary.to_string(index=False, float_format=lambda a: f"{a:.4g}"))
