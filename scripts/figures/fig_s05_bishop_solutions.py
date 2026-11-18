"""Chapter 3, Section 12.1: the simplified Bishop equation has a solution
between every two neighbouring F*, and one above the largest.

On a circle the moment balance is sum T_i = sum W_i sin(alpha_i). A slice
whose base rises gives T_i = a_i / (F_s - F_i*), which runs to plus infinity
as F_s comes down to F_i* and to minus infinity as it comes up to it; the
other slices give terms that fall smoothly as F_s grows. So between two
neighbouring F* the sum runs from plus to minus infinity and meets the level
of the right side, and above the largest F* it falls from plus infinity to
zero and meets it once.

A schematic with three rising slices: the curve is drawn to scale from POLES,
their weights A and one smooth term, none of which is printed, and the
solutions are where it meets LEVEL.
"""

from figlib import FAINT, INK, LABEL, MUTED, RESIST, RULE, SMALL, WEIGHT, Figure, L

POLES = (1.0, 1.75, 2.6)  # F* of the three rising slices
A = (0.30, 0.34, 0.50)  # their numerators over cos(alpha)
SMOOTH = (5.5, 0.8)  # the other slices: b / (F + q)
LEVEL = 2.4  # the right side, sum W sin(alpha)
F_RANGE, Y_RANGE = (0.35, 6.2), (-2.6, 6.4)
SHADE = "#eef7f2"  # where every denominator is positive
X0, X1, Y0, Y1 = 78.0, 726.0, 44.0, 278.0  # the graph, in pixels

fig = Figure(
    "fig_s05_bishop_solutions",
    358,
    L("簡易Bishop法のつり合い式を満たす安全率（模式図）",
      "The factors of safety that satisfy the simplified Bishop equation (schematic)"),
    L("底面が上るスライスが3本あるときの，底面せん断力の和（緑の曲線）と，自重による項の和（赤の破線）．"
      "底面せん断力の和は，各スライスの F* で途切れ，その上では正の大きな値に，下では負の大きな値になる．"
      "そのため，隣り合う F* の間ごとに，2つの和が等しくなる安全率がある（白丸）．"
      "この図では，最も小さい F* の下にも1つある．"
      "そこでは，少なくとも1つのスライスの分母が負である．"
      "最も大きい F* より上では，曲線は単調に減少し，2つの和は1回だけ等しくなる（黒丸）．"
      "すべてのスライスで分母が正になる解は，これだけである．",
      "The sum of the base shear forces (green curve) and the sum of the weight terms (red dashes) for three "
      "slices whose bases rise. The sum of the shear forces breaks at each slice's F*, large and positive just "
      "above it and large and negative just below. So between every two neighbouring F* there is a factor of "
      "safety at which the two sums are equal (open circles), with the denominator of at least one slice "
      "negative; here there is one below the smallest F* too. Above the largest F* the curve falls steadily and the sums are equal once (filled circle): the "
      "only solution with every denominator positive."),
)


def total(f):
    """The sum of the base shear forces at F_s = f."""
    return sum(a / (f - p) for a, p in zip(A, POLES)) + SMOOTH[0] / (f + SMOOTH[1])


def px(f):
    return X0 + (f - F_RANGE[0]) / (F_RANGE[1] - F_RANGE[0]) * (X1 - X0)


def py(y):
    return Y1 - (y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * (Y1 - Y0)


def solution(lo, hi):
    """Where the sum meets LEVEL between lo and hi, by bisection; None if it does not."""
    if (total(lo) - LEVEL) * (total(hi) - LEVEL) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (total(lo) - LEVEL) * (total(mid) - LEVEL) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


EPS = 1e-6
edges = (F_RANGE[0], *POLES, F_RANGE[1])
intervals = [(a + EPS, b - EPS) for a, b in zip(edges[:-1], edges[1:])]

# Where every denominator is positive: above the largest F*.
fig.polygon([(px(POLES[-1]), Y0), (X1, Y0), (X1, Y1), (px(POLES[-1]), Y1)], fill=SHADE)
# The two ranges are named above the graph, clear of the curve.
fig.text((0.5 * (px(POLES[-1]) + X1), Y0 - 16),
         L("すべてのスライスで分母が正", "every denominator positive"), SMALL, RESIST, "middle")
fig.text((0.5 * (X0 + px(POLES[-1])), Y0 - 16),
         L("分母が負のスライスがある", "some denominator negative"), SMALL, MUTED, "middle")

# Axes.
fig.line((X0, py(0)), (X1, py(0)), MUTED, 1.2)
fig.line((X0, Y0), (X0, Y1), MUTED, 1.2)
fig.math((X1 + 6, py(0) + 5), "F_s", 15, MUTED)
fig.text((X0 - 8, py(0) + 5), "0", SMALL, MUTED, "end")

# The F* of the three slices.
for k, pole in enumerate(POLES, start=1):
    fig.line((px(pole), Y0), (px(pole), Y1), FAINT, 1.2, "4 4")
    fig.math((px(pole), Y1 + 20), f"F^*_{k}", 15, MUTED, "middle")

# The right side.
fig.line((X0, py(LEVEL)), (X1, py(LEVEL)), WEIGHT, 1.8, "6 4")
fig.math((X1 - 4, py(LEVEL) - 8), r"\r{Σ}W_i\r{ sin }α_i", 15, WEIGHT, "end")

# The left side, one branch per interval, clipped to the graph.
for lo, hi in intervals:
    pts = []
    for j in range(241):
        f = lo + (hi - lo) * j / 240
        y = total(f)
        if Y_RANGE[0] <= y <= Y_RANGE[1]:
            pts.append((px(f), py(y)))
        elif pts:
            fig.polyline(pts, RESIST, 2.4)
            pts = []
    if len(pts) > 1:
        fig.polyline(pts, RESIST, 2.4)
fig.math((px(4.6), py(total(4.6)) + 22), r"\r{Σ}T_i", 15, RESIST, "middle")

# The solutions.
for lo, hi in intervals:
    f = solution(lo, hi)
    if f is None:
        continue
    if lo > POLES[-1]:
        fig.circle((px(f), py(LEVEL)), 6.0, fill=INK)
    else:
        fig.circle((px(f), py(LEVEL)), 5.5, fill="#ffffff", color=INK, width=2.0)

# What the two marks mean.
fig.rect(X0, 312, X1 - X0, 36, fill="none", color=RULE, width=1.2)
fig.circle((X0 + 22, 330), 5.5, fill="#ffffff", color=INK, width=2.0)
fig.text((X0 + 36, 335), L("分母が負のスライスがある解", "a solution with a negative denominator"), LABEL, INK)
fig.circle((X0 + 352, 330), 6.0, fill=INK)
fig.text((X0 + 366, 335), L("すべてのスライスで分母が正の解", "the solution with every denominator positive"),
         LABEL, INK)

if __name__ == "__main__":
    fig.save()
