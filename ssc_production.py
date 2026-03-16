"""
Sklenka Ski Company (SSC) - Daily Production Optimization
==========================================================
Linear Programming model to maximize daily net profit.

Decision Variables:
  x = number of Jordanelle ski pairs produced per day
  y = number of Deercrest ski pairs produced per day

Objective:
  Maximize Z = 50x + 65y

Constraints:
  Fabrication:  3.5x + 4.0y <= 84   (12 workers * 7 hrs)
  Finishing:    1.0x + 1.5y <= 21   (3 workers * 7 hrs)
  Demand ratio: y >= 2x              (at least 2x as many Deercrest)
  Non-neg:      x >= 0, y >= 0
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# 1. Solve with scipy.optimize.linprog (minimizes, so negate profits)
# ---------------------------------------------------------------------------
# Variables: [x, y]
c = [-50, -65]  # negate for maximization

# Inequality constraints (A_ub @ [x,y] <= b_ub)
A_ub = [
    [3.5,  4.0],   # fabrication
    [1.0,  1.5],   # finishing
    [1.0, -0.5],   # demand: -y + 2x <= 0  =>  y >= 2x  => 2x - y <= 0
]
b_ub = [84, 21, 0]

bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

x_opt = result.x[0]
y_opt = result.x[1]
z_opt = -result.fun  # negate back

# ---------------------------------------------------------------------------
# 2. Corner-point analysis (for verification)
# ---------------------------------------------------------------------------
def is_feasible(x, y, tol=1e-6):
    return (
        3.5*x + 4.0*y <= 84 + tol and
        1.0*x + 1.5*y <= 21 + tol and
        y >= 2*x - tol and
        x >= -tol and y >= -tol
    )

# Candidate corner points
corners = []

# (a) Origin
corners.append((0, 0))

# (b) x=0, finishing binding: y = 14
corners.append((0, 14))

# (c) x=0, fabrication binding: y = 21
corners.append((0, 21))

# (d) demand y=2x meets finishing 1x+1.5y=21
#     x + 1.5(2x) = 21 => 4x = 21 => x = 5.25, y = 10.5
corners.append((21/4, 21/2))

# (e) demand y=2x meets fabrication 3.5x+4y=84
#     3.5x + 8x = 84 => x = 84/11.5
x_e = 84 / 11.5
corners.append((x_e, 2*x_e))

# (f) finishing & fabrication intersection (no demand constraint)
#     3.5x + 4y = 84  and  x + 1.5y = 21
#     From finishing: x = 21 - 1.5y
#     3.5(21-1.5y) + 4y = 84 => 73.5 - 5.25y + 4y = 84 => -1.25y = 10.5 => y<0 (infeasible)
corners.append((21 - 1.5*(-8.4), -8.4))  # recorded but will be filtered

feasible_corners = [(x, y) for x, y in corners if is_feasible(x, y)]

print("=" * 60)
print("  Sklenka Ski Company – Daily Production Optimization")
print("=" * 60)
print()
print("FEASIBLE CORNER POINTS:")
print(f"  {'Point':<20} {'x (Jordanelle)':>15} {'y (Deercrest)':>14} {'Profit ($)':>12}")
print("  " + "-" * 63)
for (x, y) in feasible_corners:
    z = 50*x + 65*y
    print(f"  ({x:.4f}, {y:.4f}){'':<8} {x:>15.4f} {y:>14.4f} {z:>12.2f}")

print()
print("OPTIMAL SOLUTION (LP Solver):")
print(f"  Jordanelle pairs/day  (x) = {x_opt:.4f}")
print(f"  Deercrest  pairs/day  (y) = {y_opt:.4f}")
print(f"  Maximum daily profit  (Z) = ${z_opt:.2f}")
print()

# ---------------------------------------------------------------------------
# 3. Weekly production summary
# ---------------------------------------------------------------------------
days_per_week = 5
print("WEEKLY PRODUCTION SUMMARY (5-day week):")
print(f"  Jordanelle pairs/week = {x_opt * days_per_week:.2f}")
print(f"  Deercrest  pairs/week = {y_opt * days_per_week:.2f}")
print(f"  Total weekly profit   = ${z_opt * days_per_week:.2f}")
print()

# Verify constraints
print("CONSTRAINT VERIFICATION (daily):")
fab_used = 3.5*x_opt + 4.0*y_opt
fin_used = 1.0*x_opt + 1.5*y_opt
ratio    = y_opt / x_opt if x_opt > 0 else float("inf")
print(f"  Fabrication used : {fab_used:.2f} / 84.00 hrs  (slack: {84 - fab_used:.2f})")
print(f"  Finishing used   : {fin_used:.2f} / 21.00 hrs  (slack: {21 - fin_used:.2f})")
print(f"  Deercrest/Jordan : {ratio:.4f}  (required >= 2.00)")
print()

# ---------------------------------------------------------------------------
# 4. Graphical visualization
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

x_range = np.linspace(0, 25, 400)

# Constraint boundary lines
y_fab  = (84 - 3.5*x_range) / 4.0
y_fin  = (21 - 1.0*x_range) / 1.5
y_dem  = 2.0 * x_range

ax.plot(x_range, y_fab, label="Fabrication: 3.5x + 4y = 84", color="royalblue", lw=2)
ax.plot(x_range, y_fin, label="Finishing: x + 1.5y = 21",    color="darkorange", lw=2)
ax.plot(x_range, y_dem, label="Demand: y = 2x",               color="green",      lw=2, ls="--")

# Shade feasible region
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

feasible_pts = sorted(feasible_corners, key=lambda p: np.arctan2(p[1]-y_opt, p[0]-x_opt))
poly = Polygon(feasible_pts, closed=True, alpha=0.15, color="steelblue")
ax.add_patch(poly)

# Mark corner points
for (px, py) in feasible_corners:
    z = 50*px + 65*py
    ax.plot(px, py, "ko", ms=7, zorder=5)
    ax.annotate(f"({px:.2f}, {py:.2f})\nZ=${z:.0f}",
                xy=(px, py), xytext=(px+0.3, py+0.5),
                fontsize=8.5, color="black")

# Mark optimal point
ax.plot(x_opt, y_opt, "r*", ms=16, zorder=6, label=f"Optimal: ({x_opt:.2f}, {y_opt:.2f}), Z=${z_opt:.2f}")

ax.set_xlim(0, 22)
ax.set_ylim(0, 22)
ax.set_xlabel("Jordanelle pairs per day (x)", fontsize=12)
ax.set_ylabel("Deercrest pairs per day (y)",  fontsize=12)
ax.set_title("SSC Daily Production – Feasible Region & Optimal Solution", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ssc_production.png", dpi=150)
print("Plot saved to ssc_production.png")
