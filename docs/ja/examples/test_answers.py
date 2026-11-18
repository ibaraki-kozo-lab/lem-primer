"""確認問題の答えに示した値を確認する．

読む人がダウンロードするファイルではない．ページの答えとコードがずれないように，
CIで実行する．
"""

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

import columns
import infinite_slope
import slices

CENTRE_3D = np.array([6.0, 0.0, 18.0])
AXIS = np.array([0.0, 1.0, 0.0])
D = np.array([-1.0, 0.0, 0.0])
R = slices.Circle().radius


def base_stresses_seismic(beta_deg, w, k):
    """実践1 問3の答え：すべる向きに水平な力 k w を加えた base_stresses．"""
    n, _ = infinite_slope.plane_vectors(beta_deg)
    weight = np.array([-k * w, -w])
    area = 1.0 / math.cos(math.radians(beta_deg))
    sigma_n, shear = infinite_slope.split_traction(-weight / area, n)
    return sigma_n, float(np.linalg.norm(shear))


def test_practice_1():
    sigma_n, tau = infinite_slope.base_stresses(25.0, 18.0 * 3.0)
    assert (sigma_n, tau) == pytest.approx((44.36, 20.68), abs=0.01)
    assert infinite_slope.factor_of_safety(
        sigma_n, tau, 0.0, 10.0, 30.0
    ) == pytest.approx(1.722, abs=1e-3)

    sigma_n, tau = base_stresses_seismic(30.0, 90.0, 0.2)
    assert (sigma_n, tau) == pytest.approx((59.71, 52.47), abs=0.01)
    assert infinite_slope.factor_of_safety(
        sigma_n, tau, 0.0, 10.0, 30.0
    ) == pytest.approx(0.848, abs=1e-3)
    assert infinite_slope.factor_of_safety(
        *base_stresses_seismic(30.0, 90.0, 0.1), 0.0, 10.0, 30.0
    ) == pytest.approx(1.022, abs=1e-3)

    w = infinite_slope.column_weight(5.0, 18.0, h_w=5.0, gamma_sat=20.0)
    sigma_n, tau = infinite_slope.base_stresses(50.0, w)
    u = infinite_slope.pore_pressure(5.0, 50.0, "vertical")
    assert infinite_slope.factor_of_safety(
        sigma_n, tau, u, 10.0, 30.0
    ) == pytest.approx(0.112, abs=1e-3)
    assert infinite_slope.factor_of_safety(
        sigma_n, tau, sigma_n, 10.0, 30.0
    ) == pytest.approx(0.203, abs=1e-3)

    for z in (2.0, 5.0, 10.0):
        sigma_n, tau = infinite_slope.base_stresses(30.0, 18.0 * z)
        assert infinite_slope.factor_of_safety(
            sigma_n, tau, 0.0, 0.0, 35.0
        ) == pytest.approx(1.213, abs=1e-3)


def test_practice_2():
    s = slices.make_slices(slices.Circle(), 6)
    fs = slices.bishop(s)
    m_alpha = np.cos(s.alpha) + np.sin(s.alpha) * s.tan_phi / fs
    assert m_alpha == pytest.approx(
        [0.894, 0.987, 1.033, 1.032, 0.973, 0.822], abs=1e-3
    )

    s = slices.make_slices(slices.Circle(), 50)
    assert slices.residuals(s, slices.bishop(s), 0.0, (6.0, 18.0))[0] == pytest.approx(
        86.7, abs=0.1
    )
    assert np.sum(s.W) == pytest.approx(2415.6, abs=0.1)

    s = slices.make_slices(slices.Circle(), 50, phi_deg=25.0)
    fs, theta = slices.spencer(s, (6.0, 18.0))
    assert (
        slices.fellenius(s),
        slices.bishop(s),
        slices.janbu(s),
        fs,
    ) == pytest.approx((1.594, 1.734, 1.575, 1.731), abs=1e-3)
    assert math.degrees(theta) == pytest.approx(17.9, abs=0.1)
    # 問5：φ' = 30° のときとの比は0.84ほどで，tan φ' の比0.81より大きい
    base = slices.make_slices(slices.Circle(), 50)
    for method in (slices.fellenius, slices.bishop, slices.janbu):
        assert method(s) / method(base) == pytest.approx(0.84, abs=5e-3)
    assert fs / slices.spencer(base, (6.0, 18.0))[0] == pytest.approx(0.84, abs=5e-3)
    assert math.tan(math.radians(25.0)) / math.tan(math.radians(30.0)) == pytest.approx(
        0.81, abs=5e-3
    )

    assert slices.bishop(
        slices.make_slices(slices.Circle((4.0, 16.0)), 50)
    ) == pytest.approx(1.790, abs=1e-3)

    # 実践3 問6が引く，地下水位のある円弧の値
    s = slices.make_slices(slices.Circle(), 50, water_level=4.0)
    assert (slices.fellenius(s), slices.bishop(s)) == pytest.approx(
        (1.402, 1.540), abs=1e-3
    )
    assert np.sum(s.u * s.l) / np.sum(s.W) == pytest.approx(0.28, abs=5e-3)

    # 問9：右端のスライスの，式 eq-slices-normal の分子の2つの項と N - U
    for level, expected_fs, expected in (
        (None, 2.059, (4.36, 3.86, 0.56)),
        (4.0, 1.544, (4.40, 5.26, -0.91)),
    ):
        s = slices.make_slices(slices.Circle(), 50, water_level=level)
        fs, theta = slices.spencer(s, (6.0, 18.0))
        p = np.array([-np.sin(theta), np.cos(theta)])
        N = slices.base_forces(s, fs, theta)[0][-1]
        assert s.u[-1] == 0.0
        assert fs == pytest.approx(expected_fs, abs=1e-3)
        assert (
            s.W[-1] * p[1],
            s.c[-1] * s.l[-1] * (-s.m[-1] @ p) / fs,
            N - s.u[-1] * s.l[-1],
        ) == pytest.approx(expected, abs=0.005)


def test_practice_3():
    assert 0.25**2 / math.cos(math.radians(30.0)) == pytest.approx(0.0722, abs=1e-4)

    sphere = columns.Ellipsoid(CENTRE_3D, (R, R, R))
    wet = columns.make_columns(sphere, 0.25, water_level=4.0)
    assert columns.hovland(wet, columns.section_directions(wet, D)) == pytest.approx(
        1.418, abs=1e-3
    )
    wet_bishop = columns.bishop(wet, CENTRE_3D, AXIS)
    assert wet_bishop == pytest.approx(1.699, abs=1e-3)
    assert np.sum(wet.u * wet.A) / np.sum(wet.W) == pytest.approx(0.25, abs=5e-3)

    dry = columns.make_columns(sphere, 0.25)
    dry_bishop = columns.bishop(dry, CENTRE_3D, AXIS)
    assert dry_bishop == pytest.approx(2.142, abs=1e-3)
    # 問6：簡易Bishop法の，3次元と実践2の2次元の差
    circle = slices.Circle()
    dry_2d = slices.bishop(slices.make_slices(circle, 50))
    wet_2d = slices.bishop(slices.make_slices(circle, 50, water_level=4.0))
    assert dry_bishop - dry_2d == pytest.approx(0.079, abs=5e-4)
    assert wet_bishop - wet_2d == pytest.approx(0.159, abs=5e-4)

    raised = CENTRE_3D + np.array([0.0, 0.0, 2.0])
    assert columns.hovland(dry, columns.section_directions(dry, D)) == pytest.approx(
        1.816, abs=1e-3
    )
    assert columns.hovland_moment(dry, raised, AXIS) == pytest.approx(1.849, abs=1e-3)
    assert columns.bishop(dry, raised, AXIS) == pytest.approx(2.122, abs=1e-3)


def sign_changes(f, lo, hi, n=2000):
    """lo と hi の間で，f の符号が変わる回数．両端の少し内側から数える．"""
    x = np.geomspace(lo * (1 + 1e-9), hi * (1 - 1e-9), n)
    sign = np.sign([f(v) for v in x])
    return int(np.sum(sign[:-1] * sign[1:] < 0))


def test_chapter_3_solutions_of_the_bishop_equation():
    """第3章 12.1節：つり合い式を満たす F_s の数と，F* の値．"""
    s = slices.make_slices(slices.Circle(), 50)
    rising = s.alpha < 0.0
    f_star = np.sort(-np.tan(s.alpha[rising]) * s.tan_phi[rising])
    assert len(f_star) == 14
    assert f_star[-1] == pytest.approx(0.216, abs=1e-3)

    def balance(fs):
        m_alpha = np.cos(s.alpha) + np.sin(s.alpha) * s.tan_phi / fs
        T = (s.c * s.b + (s.W - s.u * s.b) * s.tan_phi) / (fs * m_alpha)
        return np.sum(T) - np.sum(s.W * np.sin(s.alpha))

    # F* の下，隣り合う F* の間，F* の上に，解が1つずつある
    edges = [1e-6, *f_star, 50.0]
    counts = [sign_changes(balance, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    assert counts == [1] * 15
    assert balance(slices.bishop(s)) == pytest.approx(0.0, abs=1e-6)
    assert slices.bishop(s) > f_star[-1]

    # 同じ中心と半径の球：F* の数と，隣り合う F* の間での符号の変化
    col = columns.make_columns(columns.Ellipsoid(CENTRE_3D, (R, R, R)), 0.25)
    m = columns.rotation_directions(col, AXIS)
    l_t, l_w, l_n = columns.arms(col, m, CENTRE_3D, AXIS)
    nz, mz = col.n[:, 2], m[:, 2]
    f_star = np.sort((mz * col.tan_phi / -nz)[mz > 0.0])
    # y について対称なカラムは同じ F* をもつので，同じ値をまとめる
    first = np.concatenate([[True], np.diff(f_star) > 1e-9])
    lowest, highest = f_star[first], f_star[np.roll(first, -1)]
    assert len(lowest) == 988
    assert highest[-1] == pytest.approx(0.221, abs=1e-3)

    def balance_3d(fs):
        m_alpha = -nz - mz * col.tan_phi / fs
        N = (col.W + mz * col.c * col.A / fs) / m_alpha
        T = (col.c * col.A + N * col.tan_phi) / fs
        return np.sum(T * l_t + N * l_n - col.W * l_w)

    above = np.sign([balance_3d(v * (1 + 1e-9)) for v in highest[:-1]])
    below = np.sign([balance_3d(v * (1 - 1e-9)) for v in lowest[1:]])
    assert np.all(above > 0) and np.all(below < 0)
    assert columns.bishop(col, CENTRE_3D, AXIS) > highest[-1]


def test_save_table(tmp_path, monkeypatch, capsys):
    """実践3 8節の save_table.py が，楕円体のカラムの表を保存する．"""
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(__file__).with_name("save_table.py")))
    assert capsys.readouterr().out == "4596 columns saved to columns.npz\n"
    saved = np.load(tmp_path / "columns.npz")
    assert len(saved["weight"]) == 4596
    assert np.array_equal(saved["axis"], AXIS)
