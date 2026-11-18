import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def error_plots(x, num, ref, name, xlabel, ylabel):
    dev = num - ref
    rel = np.abs(dev) / np.abs(ref) * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, dev, color="tab:red")
    ax[0].set(title="Отклонение (числ. - аналит.)", xlabel=xlabel, ylabel=ylabel)
    ax[1].plot(x, rel, color="tab:purple")
    ax[1].set(title="Относительная погрешность, %", xlabel=xlabel, ylabel="%")
    save(fig, name)
    return dev, rel


def derivative_plots(x, d1_num, d2_num, d1_an, d2_an, name, xlabel, l1, l2):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, d1_an, label="аналитика")
    ax[0].plot(x, d1_num, "--", label="конечные разности")
    ax[0].set(title=l1, xlabel=xlabel)
    ax[0].legend()
    ax[1].plot(x, d2_an, label="аналитика")
    ax[1].plot(x, d2_num, "--", label="конечные разности")
    ax[1].set(title=l2, xlabel=xlabel)
    ax[1].legend()
    save(fig, name)


def march1(A, B, C, D, L, n):
    h = L / n
    U = np.zeros(n + 1)
    U[0] = C
    U[1] = C + h * D
    for i in range(1, n):
        U[i + 1] = 2 * U[i] - U[i - 1] + h**2 * B / A
    return np.linspace(0, L, n + 1), U


def task1(A=2.0, B=4.0, C=3.0, D=1.0, L=1.0, n=50):
    h = L / n
    x, U = march1(A, B, C, D, L, n)
    U_an = B / (2 * A) * x**2 + D * x + C

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, U_an, label="аналитическое решение")
    ax.plot(x, U, "o", ms=3, label="метод конечных разностей")
    ax.set(title=f"Задание 1: A·U'' = B  (A={A}, B={B}, U(0)={C}, U'(0)={D})",
           xlabel="x", ylabel="U(x)")
    ax.legend()
    save(fig, "task1_solution.png")

    dev, rel = error_plots(x, U, U_an, "task1_error.png", "x", "U_чис - U_ан")

    d1 = np.diff(U) / h
    d2 = np.diff(U, 2) / h**2
    derivative_plots(x[:-1], d1, np.append(d2, np.nan),
                     B / A * x[:-1] + D, np.full(n, B / A),
                     "task1_derivatives.png", "x", "U'(x)", "U''(x)")

    print("=== Задание 1 ===")
    print(f"A={A}, B={B}, C={C}, D={D}, n={n}, h={h}")
    print(f"max |отклонение|        = {np.abs(dev).max():.3e}")
    print(f"max отн. погрешность, % = {rel.max():.3e}")
    print(f"U'' (числ.) = {d2.mean():.6f}, B/A = {B / A:.6f}")
    for m in (25, 50, 100, 200):
        xx, uu = march1(A, B, C, D, L, m)
        e = np.abs(uu - (B / (2 * A) * xx**2 + D * xx + C)).max()
        print(f"  n={m:4d}  max|err|={e:.3e}")


def march2(m, x0, v0, T, n):
    h = T / n
    t = np.linspace(0, T, n + 1)
    x = np.zeros(n + 1)
    x[0] = x0
    x[1] = x0 + h * v0
    a = m / h**2 + 1 / h
    for i in range(1, n):
        x[i + 1] = (t[i]**2 + x[i] * (2 * m / h**2 + 1 / h) - m * x[i - 1] / h**2) / a
    return t, x


def analytic2(m, x0, v0):
    C2 = m * (2 * m**2 - v0)
    C1 = x0 - C2

    def x_an(t):
        return C1 + C2 * np.exp(-t / m) + t**3 / 3 - m * t**2 + 2 * m**2 * t

    def v_an(t):
        return -C2 / m * np.exp(-t / m) + t**2 - 2 * m * t + 2 * m**2

    def a_an(t):
        return C2 / m**2 * np.exp(-t / m) + 2 * t - 2 * m

    return x_an, v_an, a_an


def task2(m=1.0, x0=1.0, v0=0.5, T=5.0, n=500):
    h = T / n
    t, x = march2(m, x0, v0, T, n)
    x_an, v_an, a_an = analytic2(m, x0, v0)
    xa = x_an(t)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, xa, label="аналитическое решение")
    ax.plot(t[::10], x[::10], "o", ms=3, label="метод конечных разностей")
    ax.set(title=f"Задание 2: m·x'' + x' = t²  (m={m}, x(0)={x0}, x'(0)={v0})",
           xlabel="t", ylabel="x(t)")
    ax.legend()
    save(fig, "task2_solution.png")

    dev, rel = error_plots(t, x, xa, "task2_error.png", "t", "x_чис - x_ан")

    v = np.gradient(x, h)
    acc = np.gradient(v, h)
    s = slice(2, -2)
    derivative_plots(t[s], v[s], acc[s], v_an(t[s]), a_an(t[s]),
                     "task2_derivatives.png", "t", "x'(t) — скорость", "x''(t) — ускорение")

    print("=== Задание 2 ===")
    print(f"m={m}, x(0)={x0}, x'(0)={v0}, T={T}, n={n}, h={h}")
    print(f"max |отклонение|        = {np.abs(dev).max():.3e}")
    print(f"max отн. погрешность, % = {rel.max():.3e}")
    print(f"x(T): числ.={x[-1]:.5f}  аналит.={xa[-1]:.5f}")
    for k in (125, 250, 500, 1000, 2000):
        tt, xx = march2(m, x0, v0, T, k)
        print(f"  n={k:5d}  max|err|={np.abs(xx - x_an(tt)).max():.3e}")


def solve_poisson(n, L=1.0, rhs=1.0, dtop=0.5, dright=0.5):
    h = L / n
    N = n + 1
    idx = lambda i, j: i * N + j
    M = lil_matrix((N * N, N * N))
    b = np.zeros(N * N)
    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            if j == 0 or i == 0:
                M[k, k] = 1
                b[k] = 1.0
            elif j == n:
                M[k, k] = 1
                M[k, idx(i - 1, j)] = -1
                b[k] = dtop * h
            elif i == n:
                M[k, k] = 1
                M[k, idx(i - 1, j)] = -1
                b[k] = dright * h
            else:
                M[k, idx(i + 1, j)] = 1 / h**2
                M[k, idx(i - 1, j)] = 1 / h**2
                M[k, idx(i, j + 1)] = 1 / h**2
                M[k, idx(i, j - 1)] = 1 / h**2
                M[k, k] = -4 / h**2
                b[k] = rhs
    return spsolve(M.tocsr(), b).reshape(N, N)


def task3(n=50, L=1.0):
    h = L / n
    U = solve_poisson(n, L)
    x = np.linspace(0, L, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, U, cmap="viridis", linewidth=0, antialiased=True)
    ax.set(title="Задание 3: U(x, y)", xlabel="x", ylabel="y", zlabel="U")
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(X, Y, U, 20, cmap="viridis")
    ax2.contour(X, Y, U, 20, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax2, label="U")
    ax2.set(title="Линии уровня", xlabel="x", ylabel="y", aspect="equal")
    save(fig, "task3_solution.png")

    lap = (U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]) / h**2
    print("=== Задание 3 ===")
    print(f"n={n}, h={h}")
    print(f"min U = {U.min():.5f}, max U = {U.max():.5f}")
    print(f"U(0.5, 0.5) = {U[n // 2, n // 2]:.6f}")
    print(f"невязка ΔU - 1 (макс.) = {np.abs(lap - 1).max():.2e}")
    print(f"U_x(1, y=0.5) = {(U[n, n // 2] - U[n - 1, n // 2]) / h:.4f}")
    print(f"U(1, 1) = {U[n, n]:.4f}")
    for m in (10, 20, 40, 80, 160):
        Um = solve_poisson(m)
        print(f"  n={m:4d}  U(0.5,0.5)={Um[m // 2, m // 2]:.6f}")


if __name__ == "__main__":
    task1()
    task2()
    task3()
