"""Clausius-Clapeyron ODE (Problem 2, vapour pressure).

dP/dT = L P / (R T^2), T in kelvin, L constant.
Analytic solution: P = P0 exp[-(L/R)(1/T - 1/T0)].
"""
import numpy as np

R = 8.314462618  # J / (mol K)


def f(T, P, L):
    return L * P / (R * T**2)


def euler(T0, P0, T_end, n, L):
    h = (T_end - T0) / n
    T = T0 + h * np.arange(n + 1)
    P = np.empty(n + 1)
    P[0] = P0
    for i in range(n):
        P[i + 1] = P[i] + h * f(T[i], P[i], L)
    return T, P


def rk4(T0, P0, T_end, n, L):
    h = (T_end - T0) / n
    T = T0 + h * np.arange(n + 1)
    P = np.empty(n + 1)
    P[0] = P0
    for i in range(n):
        k1 = f(T[i], P[i], L)
        k2 = f(T[i] + h / 2, P[i] + h / 2 * k1, L)
        k3 = f(T[i] + h / 2, P[i] + h / 2 * k2, L)
        k4 = f(T[i] + h, P[i] + h * k3, L)
        P[i + 1] = P[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return T, P


def analytic(T, P0, T0, L):
    return P0 * np.exp(-(L / R) * (1.0 / T - 1.0 / T0))


def invariant(T, P, L):
    """I = ln P + L/(R T) is constant along the exact solution."""
    return np.log(P) + L / (R * T)


def measured_order(h, err, floor=1e-12, hmax=5.0):
    """Slope of log(err) vs log(h) in the asymptotic, pre-round-off region."""
    h, err = np.asarray(h), np.asarray(err)
    m = (err > floor) & (h <= hmax)
    return np.polyfit(np.log(h[m]), np.log(err[m]), 1)[0]
