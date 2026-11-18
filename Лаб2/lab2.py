"""Лабораторная работа 2. Изучение броуновского движения.

Проверка формулы Эйнштейна <ΔS²> = 4·D·t по координатам одной частицы,
снятым через равные промежутки времени τ.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ЛР 2", "tablitsa_pyltsa.xls")
OUT = os.path.join(HERE, "plots")
XLSX = os.path.join(HERE, "Результаты_ЛР2.xlsx")
os.makedirs(OUT, exist_ok=True)

TAU = 4.0          # шаг по времени между кадрами, с (из шапки таблицы: dt = 4 s)
K_MAX = 9          # t = τ, 2τ, …, 9τ (столбцы таблицы)
# Масштаб изображения, мкм на пиксель. Если преподаватель даст значение,
# впишите его сюда — программа пересчитает D в мкм²/с и м²/с.
SCALE_UM_PER_PX = None

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def load_coords(path):
    """Читает координаты x, y (пиксели) с первого листа, пока они заполнены."""
    sh = xlrd.open_workbook(path).sheet_by_index(0)
    pts = []
    for r in range(2, sh.nrows):
        x, y = sh.cell_value(r, 1), sh.cell_value(r, 2)
        if x == "" or y == "":
            break
        pts.append((float(x), float(y)))
    return np.array(pts)


def msd(xy, k_max):
    """Для каждого k: все квадраты смещений (ΔS)² между точками i и i+k (как в Таблице 1)."""
    sq = []
    for k in range(1, k_max + 1):
        d = xy[k:] - xy[:-k]
        sq.append((d**2).sum(axis=1))
    return sq


def fit_through_zero(t, s2):
    """МНК для прямой <ΔS²> = a·t (через начало координат, как в формуле Эйнштейна)."""
    a = (t * s2).sum() / (t * t).sum()
    resid = s2 - a * t
    da = np.sqrt((resid**2).sum() / (len(t) - 1) / (t * t).sum())
    return a, da


def plot_trajectory(xy):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(xy[:, 0], xy[:, 1], "-o", ms=3, lw=0.8, color="tab:blue")
    ax.plot(*xy[0], "o", ms=9, color="tab:green", label="начало (точка 1)")
    ax.plot(*xy[-1], "s", ms=9, color="tab:red", label=f"конец (точка {len(xy)})")
    ax.set(title="Траектория броуновской частицы", xlabel="x, пиксели", ylabel="y, пиксели", aspect="equal")
    ax.invert_yaxis()  # у изображения ось y направлена вниз
    ax.legend()
    save(fig, "trajectory.png")


def plot_msd(t, s2, err, a, da, a2, b2):
    tt = np.linspace(0, t.max() * 1.05, 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(t, s2, yerr=err, fmt="o", capsize=4, color="tab:blue", label="эксперимент ⟨ΔS²⟩")
    ax.plot(tt, a * tt, color="tab:red", label=f"⟨ΔS²⟩ = a·t,  a = {a:.0f} ± {da:.0f} пикс²/с")
    ax.plot(tt, a2 * tt + b2, "--", color="tab:gray", label=f"⟨ΔS²⟩ = a·t + b,  a = {a2:.0f}, b = {b2:.0f}")
    ax.set(title="Зависимость среднего квадрата смещения от времени",
           xlabel="t, с", ylabel="⟨ΔS²⟩, пикс²", xlim=(0, None), ylim=(0, None))
    ax.legend()
    save(fig, "msd.png")


def plot_steps(xy):
    d = np.diff(xy, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, name in enumerate(("Δx", "Δy")):
        v = d[:, i]
        sig = v.std(ddof=1)
        ax[i].hist(v, bins=12, density=True, alpha=0.6, color="tab:blue", edgecolor="k", label="эксперимент")
        g = np.linspace(v.min() - 10, v.max() + 10, 200)
        ax[i].plot(g, np.exp(-(g - v.mean())**2 / (2 * sig**2)) / (sig * np.sqrt(2 * np.pi)),
                   color="tab:red", label=f"Гаусс, σ = {sig:.1f}")
        ax[i].set(title=f"Распределение смещений {name} за τ = {TAU:.0f} с", xlabel=f"{name}, пиксели")
        ax[i].legend()
    save(fig, "steps_hist.png")


def write_xlsx(xy, sq, t, s2, a, da, D):
    """Заполненная таблица в виде Таблицы 1 из методички + лист с итогами."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Частица 1"
    bold = Font(bold=True)
    c = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws["A1"], ws["B1"], ws["D1"] = "№", "Координаты частицы", f"(ΔS)², пикс²   (τ = {TAU:.0f} с)"
    ws.merge_cells("B1:C1")
    ws.merge_cells(start_row=1, start_column=4, end_row=1, end_column=3 + K_MAX)
    ws["B2"], ws["C2"] = "x, пикс", "y, пикс"
    for k in range(1, K_MAX + 1):
        ws.cell(2, 3 + k, f"пара i–(i+{k})\nt = {k}τ = {k * TAU:.0f} с")
    n = len(xy)
    first, last = 3, 2 + n
    for i in range(n):
        r = first + i
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, xy[i, 0])
        ws.cell(r, 3, xy[i, 1])
        for k in range(1, K_MAX + 1):
            if i + k < n:
                # формула, чтобы в Excel было видно, как считается каждое число
                ws.cell(r, 3 + k, f"=(B{r + k}-B{r})^2+(C{r + k}-C{r})^2")
    r = last + 2
    labels = ("Σ(ΔS)²", "число пар n−k", "⟨ΔS²⟩ = Σ/(n−k)", "t, с")
    for j, lab in enumerate(labels):
        ws.cell(r + j, 1, lab).font = bold
        ws.merge_cells(start_row=r + j, start_column=1, end_row=r + j, end_column=3)
    for k in range(1, K_MAX + 1):
        col = get_column_letter(3 + k)
        ws.cell(r, 3 + k, f"=SUM({col}{first}:{col}{last - k})")
        ws.cell(r + 1, 3 + k, n - k)
        ws.cell(r + 2, 3 + k, f"={col}{r}/{col}{r + 1}").number_format = "0.0"
        ws.cell(r + 3, 3 + k, k * TAU)
    for row in ws.iter_rows(min_row=1, max_row=2):
        for cell in row:
            cell.font, cell.alignment = bold, c
    ws.row_dimensions[2].height = 32
    ws.column_dimensions["A"].width = 8
    for k in range(2, 4 + K_MAX):
        ws.column_dimensions[get_column_letter(k)].width = 13

    ws2 = wb.create_sheet("Итог")
    ws2.append(["t, с", "⟨ΔS²⟩, пикс²", "число пар", "a·t (прямая), пикс²"])
    for k in range(K_MAX):
        ws2.append([float(t[k]), round(float(s2[k]), 1), len(sq[k]), round(float(a * t[k]), 1)])
    ws2.append([])
    ws2.append(["Наклон a, пикс²/с", round(a, 1), "±", round(da, 1)])
    ws2.append(["D = a/4, пикс²/с", round(D, 1), "±", round(da / 4, 1)])
    for cell in ws2[1]:
        cell.font = bold
    for col in "ABCD":
        ws2.column_dimensions[col].width = 22
    wb.save(XLSX)


def main():
    xy = load_coords(SRC)
    n = len(xy)
    sq = msd(xy, K_MAX)
    k = np.arange(1, K_MAX + 1)
    t = k * TAU
    s2 = np.array([v.mean() for v in sq])
    err = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in sq])

    a, da = fit_through_zero(t, s2)
    a2, b2 = np.polyfit(t, s2, 1)
    r = np.corrcoef(t, s2)[0, 1]
    D, dD = a / 4, da / 4

    plot_trajectory(xy)
    plot_msd(t, s2, err, a, da, a2, b2)
    plot_steps(xy)
    write_xlsx(xy, sq, t, s2, a, da, D)

    print(f"Точек: n = {n}, τ = {TAU} с, полное время наблюдения {(n - 1) * TAU:.0f} с")
    print(" k   t,с  пар   <ΔS²>,пикс²   ±σ")
    for i in range(K_MAX):
        print(f"{k[i]:2d}  {t[i]:4.0f}  {len(sq[i]):3d}  {s2[i]:10.1f}  {err[i]:7.1f}")
    print(f"Прямая через 0:  a = {a:.1f} ± {da:.1f} пикс²/с  ->  D = a/4 = {D:.1f} ± {dD:.1f} пикс²/с")
    print(f"Прямая a·t+b:    a = {a2:.1f} пикс²/с, b = {b2:.0f} пикс²  ->  D = {a2 / 4:.1f} пикс²/с")
    print(f"Коэффициент корреляции r = {r:.4f}")
    d1 = np.diff(xy, axis=0)
    print(f"<Δx²>(τ) = {(d1[:, 0]**2).mean():.0f}, <Δy²>(τ) = {(d1[:, 1]**2).mean():.0f} пикс²")
    print(f"Средний шаг: <Δx> = {d1[:, 0].mean():.2f}, <Δy> = {d1[:, 1].mean():.2f} пикс")
    if SCALE_UM_PER_PX:
        Dum = D * SCALE_UM_PER_PX**2
        print(f"D = {Dum:.3f} мкм²/с = {Dum * 1e-12:.2e} м²/с")
        kT, eta = 1.380649e-23 * 293, 1.0e-3
        print(f"Радиус частицы по Стоксу–Эйнштейну a = kT/(6πηD) = {kT / (6 * np.pi * eta * Dum * 1e-12) * 1e6:.2f} мкм")


if __name__ == "__main__":
    main()
