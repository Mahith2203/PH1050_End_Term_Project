"""Braking ODE (Problem 2, stopping distance).

State y = [x, v] with dy/dt = [v, -mu*g].  Analytic braking distance is
v0**2 / (2 mu g) and the total stopping distance adds v0 * t_r.
"""
import numpy as np

G = 9.81


def analytic_braking_distance(v0, mu=0.7, g=G):
    return v0**2 / (2.0 * mu * g)


def analytic_stopping_distance(v0, t_r=0.7, mu=0.7, g=G):
    return v0 * t_r + analytic_braking_distance(v0, mu, g)


def euler(dt, v0=20.0, mu=0.7, g=G):
    """Forward Euler; the last step is shortened so the car stops exactly."""
    a = mu * g
    x, v, t = 0.0, v0, 0.0
    time, position, velocity = [t], [x], [v]
    while v > 1e-12:
        h = min(dt, v / a)
        x = x + v * h
        v = max(0.0, v - a * h)
        t += h
        time.append(t); position.append(x); velocity.append(v)
    return np.array(time), np.array(position), np.array(velocity)


def _f(y, a):
    return np.array([y[1], -a])


def rk4(dt, v0=20.0, mu=0.7, g=G):
    """Classical fourth-order Runge-Kutta with the same stopping logic."""
    a = mu * g
    y = np.array([0.0, v0])
    t = 0.0
    time, position, velocity = [t], [y[0]], [y[1]]
    while y[1] > 1e-12:
        h = min(dt, y[1] / a)
        k1 = _f(y, a)
        k2 = _f(y + h * k1 / 2, a)
        k3 = _f(y + h * k2 / 2, a)
        k4 = _f(y + h * k3, a)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[1] = max(0.0, y[1])
        t += h
        time.append(t); position.append(y[0]); velocity.append(y[1])
    return np.array(time), np.array(position), np.array(velocity)


def energy(position, velocity, m=1200.0, mu=0.7, g=G):
    """Kinetic energy plus work done by friction; constant for exact dynamics."""
    return 0.5 * m * velocity**2 + mu * m * g * position
