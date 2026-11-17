"""Regenerates the stopping-distance verification figures and tables.

Run from the repository root:  python src/braking_verification.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import braking as B

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

m, mu, g, t_r, v0, dt = 1200.0, 0.7, B.G, 0.7, 20.0, 0.01
a = mu * g
d_brake = B.analytic_braking_distance(v0, mu)
t_stop = v0 / a

te, xe, ve = B.euler(dt); tk, xk, vk = B.rk4(dt)
E0 = 0.5 * m * v0**2
Ee, Ek = B.energy(xe, ve), B.energy(xk, vk)

print(f"braking distance   analytic {d_brake:.6f}  Euler {xe[-1]:.6f}  RK4 {xk[-1]:.6f}")
print(f"total stopping     analytic {B.analytic_stopping_distance(v0):.6f}")
print(f"energy drift       Euler {np.abs(Ee - E0).max() / E0:.3e}  RK4 {np.abs(Ek - E0).max() / E0:.3e}")

dts = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
err_e = np.array([abs(d_brake - B.euler(d)[1][-1]) for d in dts])
err_k = np.array([abs(d_brake - B.rk4(d)[1][-1]) for d in dts])
order_e = np.polyfit(np.log(dts), np.log(err_e), 1)[0]
print(f"Euler measured order {order_e:.3f}; RK4 max error {err_k.max():.2e} (round-off)")
pd.DataFrame({"dt_s": dts, "euler_error_m": err_e, "rk4_error_m": err_k}).to_csv(
    TAB / "stopping_convergence_table.csv", index=False)

tx = np.linspace(0, t_stop, 200)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(tx, v0 * tx - 0.5 * a * tx**2, "k", lw=2, label="Analytic")
ax[0].plot(te, xe, "--", label="Euler"); ax[0].plot(tk, xk, ":", lw=2.5, label="RK4")
ax[0].set(xlabel="Time (s)", ylabel="Braking distance (m)", title="Position vs time"); ax[0].legend()
ax[1].plot(te, Ee / 1e3, label="Euler"); ax[1].plot(tk, Ek / 1e3, "--", label="RK4")
ax[1].axhline(E0 / 1e3, color="k", ls=":", label="Initial energy")
ax[1].set(xlabel="Time (s)", ylabel="KE + friction work (kJ)", title="Energy check"); ax[1].legend()
ax[2].loglog(dts, err_e, "o-", label=f"Euler (order {order_e:.2f})")
ax[2].loglog(dts, np.maximum(err_k, 1e-17), "s--", label="RK4 (round-off level)")
ax[2].set(xlabel="Time step (s)", ylabel="Error in braking distance (m)", title="Convergence"); ax[2].legend()
for x in ax: x.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "stopping_verification_panels.png", dpi=150); plt.close(fig)
