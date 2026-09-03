#!/usr/bin/env python3
"""Reproducible pedagogical experiments for the McCulloch student guide.

The program produces every quantitative/schematic figure in PDF, PNG, and SVG
plus a machine-readable JSON record.  It does not digitize historical panels or
fit the 1959 frog recordings.  The retinal section is a deliberately simple
normalized LN/LNP-style realization of the four principal qualitative
operations reported by Lettvin, Maturana, McCulloch, and Pitts. The rare fifth
absolute-darkness class mentioned in the paper is discussed in the text but is
not promoted to a fifth modeled channel.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter, shift, zoom


SEED = 1943
COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "cyan": "#55A6C1",
    "orange": "#E07A3F",
    "red": "#B6423C",
    "green": "#4B8B6A",
    "gold": "#D2A93B",
    "purple": "#76538B",
    "gray": "#66717E",
    "light": "#EDF2F5",
}


@dataclass
class GateResult:
    gate: str
    truth_table: list[list[int]]
    exact: bool


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def mp_unit(excitatory: np.ndarray, inhibitory: np.ndarray | None, theta: int) -> int:
    """Original-style threshold unit: any active inhibitory input vetoes firing."""
    inhibited = inhibitory is not None and bool(np.any(inhibitory))
    return int((not inhibited) and int(np.sum(excitatory)) >= theta)


def weighted_unit(x: np.ndarray, w: np.ndarray, theta: float) -> int:
    return int(float(np.dot(x, w)) >= theta)


def figure_timeline(outdir: Path) -> dict:
    mcculloch = [
        (1898, "born", "life"),
        (1927, "M.D.", "life"),
        (1943, "logical calculus", "logic"),
        (1945, "heterarchy", "logic"),
        (1948, "Hixon lecture", "cybernetics"),
        (1952, "MIT RLE", "life"),
        (1959, "frog retina", "experiment"),
        (1965, "Embodiments of Mind", "cybernetics"),
        (1969, "dies", "life"),
    ]
    pitts = [
        (1923, "born", "life"),
        (1941, "meets McCulloch", "life"),
        (1943, "logical calculus; MIT", "logic"),
        (1947, "universals", "invariance"),
        (1952, "MIT RLE", "life"),
        (1959, "frog retina", "experiment"),
        (1969, "dies", "life"),
    ]
    cmap = {"life": COLORS["gray"], "logic": COLORS["blue"], "invariance": COLORS["purple"], "cybernetics": COLORS["orange"], "experiment": COLORS["green"]}
    fig, ax = plt.subplots(figsize=(11.4, 3.25))
    tracks = [(0.52, "Warren McCulloch (1898--1969)", mcculloch), (-0.52, "Walter Pitts (1923--1969)", pitts)]
    label_positions = {
        ("Warren McCulloch (1898--1969)", 1943): (1940.8, 0.31),
        ("Warren McCulloch (1898--1969)", 1945): (1944.7, 0.54),
        ("Warren McCulloch (1898--1969)", 1948): (1949.6, 0.31),
        ("Warren McCulloch (1898--1969)", 1952): (1954.2, 0.54),
        ("Warren McCulloch (1898--1969)", 1959): (1958.0, 0.54),
        ("Warren McCulloch (1898--1969)", 1965): (1965.0, 0.31),
        ("Warren McCulloch (1898--1969)", 1969): (1969.8, 0.54),
        ("Walter Pitts (1923--1969)", 1941): (1938.9, -0.31),
        ("Walter Pitts (1923--1969)", 1943): (1944.1, -0.54),
        ("Walter Pitts (1923--1969)", 1947): (1949.0, -0.31),
        ("Walter Pitts (1923--1969)", 1952): (1954.5, -0.54),
    }
    for y0, name, events in tracks:
        first_year = min(y for y, _, _ in events)
        ax.hlines(y0, first_year, 1969, color=COLORS["navy"], lw=1.8)
        ax.text(first_year - 1.4, y0, name, ha="right", va="center", color=COLORS["navy"], weight="bold", fontsize=8.6)
        for i, (year, label, group) in enumerate(events):
            dy = 0.29 if y0 > 0 else -0.29
            text_x, dy = label_positions.get((name, year), (year, dy))
            ax.plot([year, text_x], [y0, y0 + 0.62 * dy], color=cmap[group], lw=1.15)
            ax.scatter([year], [y0], s=46, color=cmap[group], edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(text_x, y0 + dy, f"{year}\n{label}", ha="center", va="center", color=COLORS["navy"], fontsize=7.15)
    ax.set_xlim(1893, 1972)
    ax.set_ylim(-1.24, 1.24)
    ax.axis("off")
    ax.set_title("Two unequal lives, one collaborative program", color=COLORS["navy"], weight="bold")
    save_figure(fig, outdir, "fig01_historical_timeline")
    return {
        "mcculloch_events": [{"year": y, "label": label} for y, label, _ in mcculloch],
        "pitts_events": [{"year": y, "label": label} for y, label, _ in pitts],
        "hixon_lecture_delivered": 1948,
        "hixon_volume_published": 1951,
    }


def gate_truth_tables() -> list[GateResult]:
    rows = []
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        and_y = mp_unit(np.array([a, b]), None, 2)
        or_y = mp_unit(np.array([a, b]), None, 1)
        not_a = mp_unit(np.array([1]), np.array([a]), 1)
        # XOR as (a OR b) AND NOT(a AND b), requiring two logical stages.
        any_ab = or_y
        both_ab = and_y
        xor_y = mp_unit(np.array([any_ab]), np.array([both_ab]), 1)
        rows.append([a, b, and_y, or_y, not_a, xor_y])
    names = ["AND", "OR", "NOT A", "XOR"]
    cols = [2, 3, 4, 5]
    expected = {
        "AND": [0, 0, 0, 1],
        "OR": [0, 1, 1, 1],
        "NOT A": [1, 1, 0, 0],
        "XOR": [0, 1, 1, 0],
    }
    return [GateResult(n, [[r[0], r[1], r[c]] for r in rows], [r[c] for r in rows] == expected[n]) for n, c in zip(names, cols)]


def draw_node(ax, xy, label, color=COLORS["blue"], radius=0.12):
    c = Circle(xy, radius, facecolor="white", edgecolor=color, lw=1.8, zorder=3)
    ax.add_patch(c)
    ax.text(*xy, label, ha="center", va="center", color=COLORS["navy"], fontsize=8, zorder=4)


def arrow(ax, start, end, color=COLORS["gray"], inhibitory=False, rad=0.0):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=1.2, color=color, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(patch)
    if inhibitory:
        # Overlay a short veto bar near the target.
        x, y = end
        ax.plot([x - 0.025, x + 0.025], [y - 0.035, y + 0.035], color=color, lw=1.6, zorder=5)


def figure_logic(outdir: Path) -> dict:
    gates = gate_truth_tables()
    fig = plt.figure(figsize=(10.8, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95], wspace=0.35, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A  Original inhibitory-veto unit", loc="left", weight="bold")
    draw_node(ax, (0.75, 0.5), r"$x_i(t+1)$")
    for y, lab in [(0.76, r"$e_1$"), (0.52, r"$e_2$"), (0.24, r"$h$")]:
        draw_node(ax, (0.15, y), lab, COLORS["green"] if lab != r"$h$" else COLORS["red"], radius=0.09)
    arrow(ax, (0.24, 0.76), (0.63, 0.56), COLORS["green"])
    arrow(ax, (0.24, 0.52), (0.63, 0.51), COLORS["green"])
    arrow(ax, (0.24, 0.24), (0.63, 0.44), COLORS["red"], inhibitory=True)
    ax.text(0.5, 0.06, r"fires if $e_1+e_2\geq\theta$ and $h=0$", ha="center", fontsize=8.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B  One unit cannot compute XOR", loc="left", weight="bold")
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    ax.scatter(pts[labels == 0, 0], pts[labels == 0, 1], marker="o", s=80, facecolor="white", edgecolor=COLORS["blue"], lw=2, label="XOR = 0")
    ax.scatter(pts[labels == 1, 0], pts[labels == 1, 1], marker="s", s=70, color=COLORS["orange"], label="XOR = 1")
    xx = np.linspace(-0.15, 1.15, 100)
    ax.plot(xx, 0.5 - xx, "--", color=COLORS["gray"], lw=1)
    ax.plot(xx, 1.5 - xx, "--", color=COLORS["gray"], lw=1)
    ax.text(0.5, -0.20, "no single separating line", ha="center", color=COLORS["red"], fontsize=8.5)
    ax.set(xlim=(-0.2, 1.2), ylim=(-0.27, 1.2), xticks=[0, 1], yticks=[0, 1], xlabel="$a$", ylabel="$b$")
    # Legend placed below the axes: inside the panel its marker keys read as two
    # extra data points and sat on top of the candidate separating lines.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7.5,
        handletextpad=0.5, columnspacing=1.6, borderpad=0.2,
    )

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("C  Two-stage XOR network", loc="left", weight="bold")
    for xy, lab in [((0.10, 0.70), "$a$"), ((0.10, 0.30), "$b$")]: draw_node(ax, xy, lab, COLORS["green"], 0.08)
    draw_node(ax, (0.49, 0.73), "OR", COLORS["blue"], 0.10)
    draw_node(ax, (0.49, 0.27), "AND", COLORS["blue"], 0.10)
    draw_node(ax, (0.86, 0.50), "XOR", COLORS["orange"], 0.11)
    for y0 in (0.70, 0.30):
        arrow(ax, (0.18, y0), (0.39, 0.73), COLORS["green"])
        arrow(ax, (0.18, y0), (0.39, 0.27), COLORS["green"])
    arrow(ax, (0.59, 0.73), (0.75, 0.54), COLORS["green"])
    arrow(ax, (0.59, 0.27), (0.75, 0.46), COLORS["red"], inhibitory=True)
    ax.text(0.5, 0.04, "unit delays make logical depth physical time", ha="center", fontsize=8.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    col_labels = ["$a$", "$b$", "AND", "OR", "NOT $a$", "XOR (2 stages)"]
    rows = []
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        rows.append([a, b, a & b, a | b, 1 - a, a ^ b])
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center", colLoc="center", bbox=[0.08, 0.12, 0.84, 0.78])
    table.auto_set_font_size(False); table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(COLORS["navy"] if r == 0 else (COLORS["light"] if r % 2 else "white"))
        if r == 0: cell.set_text_props(color="white", weight="bold")
    save_figure(fig, outdir, "fig02_threshold_logic")
    return {"gates": [asdict(g) for g in gates], "xor_single_unit_linearly_separable": False, "xor_network_depth": 2}


def finite_dynamics() -> tuple[dict[int, int], list[list[int]], dict[int, dict]]:
    """Three-bit network: F(x,y,z)=(y,z,x and not y), with transients."""
    transition = {}
    for s in range(8):
        x, y, z = (s >> 2) & 1, (s >> 1) & 1, s & 1
        nxt = (y << 2) | (z << 1) | (x & (1 - y))
        transition[s] = nxt
    cycles_by_key: dict[tuple[int, ...], list[int]] = {}
    orbit_data = {}
    for s in range(8):
        first_seen, path, u = {}, [], s
        while u not in first_seen:
            first_seen[u] = len(path)
            path.append(u)
            u = transition[u]
        mu = first_seen[u]
        cycle = path[mu:]
        rotations = [tuple(cycle[k:] + cycle[:k]) for k in range(len(cycle))]
        key = min(rotations)
        cycles_by_key[key] = list(key)
        orbit_data[s] = {"mu": mu, "lambda": len(cycle), "path": path, "cycle": list(key)}
    return transition, list(cycles_by_key.values()), orbit_data


def figure_dynamics_heterarchy(outdir: Path) -> dict:
    transition, cycles, orbit_data = finite_dynamics()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))

    ax = axes[0]
    ax.set_title("A  Feedback makes a state machine", loc="left", weight="bold", pad=18)
    pos = {s: (math.cos(2 * math.pi * s / 8), math.sin(2 * math.pi * s / 8)) for s in range(8)}
    cyclic_states = {s for cyc in cycles for s in cyc}
    for s, nxt in transition.items():
        if s == nxt:
            # Fixed point: draw a real self-loop anchored on the node boundary and
            # bulging radially outward. The previous stub was shorter than the node
            # radius, so it read as a stray arrowhead rather than a 000 -> 000 edge.
            px, py = pos[s]
            r_node = 0.17
            ang = math.atan2(py, px)
            spread = math.radians(50)
            start = (px + r_node * math.cos(ang - spread), py + r_node * math.sin(ang - spread))
            end = (px + r_node * math.cos(ang + spread), py + r_node * math.sin(ang + spread))
            loop = FancyArrowPatch(
                start, end,
                arrowstyle="-|>", mutation_scale=10, lw=1.15,
                color=COLORS["gray"], connectionstyle="arc3,rad=1.9",
                zorder=4,
            )
            ax.add_patch(loop)
        else:
            arrow(ax, np.array(pos[s]) * 0.88, np.array(pos[nxt]) * 0.88, COLORS["gray"], rad=0.15)
    for s, xy in pos.items():
        bits = f"{s:03b}"
        draw_node(ax, xy, bits, COLORS["blue"] if s in cyclic_states else COLORS["purple"], 0.17)
    ax.text(0, -1.35, r"$F(x,y,z)=(y,z,x\wedge\neg y)$", ha="center", fontsize=9)
    ax.set_xlim(-1.42, 1.58); ax.set_ylim(-1.5, 1.25); ax.axis("off"); ax.set_aspect("equal")

    ax = axes[1]
    ax.set_title("B  Eventually periodic is inevitable", loc="left", weight="bold", pad=18)
    t = np.arange(13)
    x = np.zeros((3, len(t)), dtype=int)
    x[:, 0] = [1, 0, 1]
    for k in range(len(t) - 1):
        x[:, k + 1] = [x[1, k], x[2, k], x[0, k] & (1 - x[1, k])]
    for i, (lab, col) in enumerate(zip(["$x$", "$y$", "$z$"], [COLORS["blue"], COLORS["orange"], COLORS["green"]])):
        ax.step(t, x[i] + 1.25 * (2 - i), where="post", label=lab, color=col, lw=1.8)
    # Headroom above the top ($x$) trace holds the regime labels; the legend moves
    # outside the axes entirely so it no longer sits on the $x$ trace.
    ax.set(xlabel="discrete time", yticks=[0, 1, 1.25, 2.25, 2.5, 3.5], yticklabels=[],
           xlim=(0, 12), ylim=(-0.25, 4.45))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5,
              handletextpad=0.5, columnspacing=1.8, borderpad=0.2)
    # Shading/guide stop below the annotation strip (traces top out at y=3.5).
    ax.axvspan(0, 3, ymin=0.0, ymax=0.82, color=COLORS["purple"], alpha=0.09)
    ax.axvline(3, ymin=0.0, ymax=0.82, color=COLORS["gray"], ls="--", lw=1)
    # Previously at y=-0.52, where they collided with the x tick labels and xlabel.
    # The transient label is left-anchored just inside the y axis: centring it on the
    # transient region made it overhang the spine, which cut through the word.
    ax.text(0.25, 3.95, r"transient $\mu=3$", ha="left", va="center", color=COLORS["purple"], fontsize=8.3)
    ax.text(7.5, 3.95, r"cycle $\lambda=3$", ha="center", va="center", color=COLORS["blue"], fontsize=8.3)

    ax = axes[2]
    ax.set_title("C  Heterarchy forbids one value scale", loc="left", weight="bold", pad=18)
    prefs = [("A", "B"), ("B", "C"), ("C", "A")]
    p = {"A": (0, 0.88), "B": (-0.82, -0.48), "C": (0.82, -0.48)}
    for a, b in prefs: arrow(ax, p[b], p[a], COLORS["red"], rad=0.12)  # arrow means a preferred to b
    for n, xy in p.items(): draw_node(ax, xy, n, COLORS["orange"], 0.17)
    ax.text(0, -1.05, r"$A\succ B\succ C\succ A$", ha="center", fontsize=10, color=COLORS["red"])
    ax.text(0, -1.34, r"no scalar $U$: summing $U(A)>U(B)>U(C)>U(A)$ is impossible", ha="center", fontsize=7.8)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.48, 1.25); ax.axis("off"); ax.set_aspect("equal")
    save_figure(fig, outdir, "fig03_feedback_heterarchy")
    return {
        "transition": {f"{k:03b}": f"{v:03b}" for k, v in transition.items()},
        "cycles": [[f"{s:03b}" for s in c] for c in cycles],
        "orbits": {f"{s:03b}": data for s, data in orbit_data.items()},
        "example_initial_state": "101",
        "example_transient_length": orbit_data[5]["mu"],
        "example_cycle_length": orbit_data[5]["lambda"],
        "finite_state_bound": 8,
        "preference_cycle_has_scalar_utility": False,
    }


def base_shape(n: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    body = ((xx / 0.42) ** 2 + (yy / 0.23) ** 2) < 1
    notch = ((xx + 0.17) ** 2 + (yy - 0.05) ** 2) < 0.055
    tail = (xx > 0.25) & (xx < 0.66) & (np.abs(yy + 0.04) < 0.07 + 0.12 * (xx - 0.25))
    return (body | tail) & (~notch)


def transform_image(img: np.ndarray, scale: float, dx: float, dy: float) -> np.ndarray:
    n = img.shape[0]
    z = zoom(img.astype(float), scale, order=1)
    canvas = np.zeros_like(img, dtype=float)
    if z.shape[0] >= n:
        a = (z.shape[0] - n) // 2
        z = z[a:a+n, a:a+n]
    else:
        a = (n - z.shape[0]) // 2
        canvas[a:a+z.shape[0], a:a+z.shape[1]] = z
        z = canvas
    return shift(z, shift=(dy, dx), order=1, mode="constant", cval=0.0)


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    return float(np.dot(av, bv) / max(na * nb, 1e-12))


def figure_universals(outdir: Path) -> dict:
    template = base_shape()
    scales = [0.72, 0.86, 1.0, 1.16, 1.32]
    translations = [(dx, dy) for dx in (-10, 0, 10) for dy in (-8, 0, 8)]
    exemplars = [transform_image(template, s, dx, dy) for s in scales for dx, dy in translations]
    probe_scales = np.linspace(0.68, 1.36, 21)
    probe = [transform_image(template, s, 8 * math.sin(2*s), -7 * math.cos(3*s)) for s in probe_scales]
    direct = np.array([normalized_corr(template, x) for x in probe])
    pooled = np.array([max(normalized_corr(e, x) for e in exemplars) for x in probe])

    fig = plt.figure(figsize=(11.1, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.4, wspace=0.30)
    for j, (s, dx, dy) in enumerate([(0.72, -10, 5), (1.0, 0, 0), (1.30, 9, -6)]):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(transform_image(template, s, dx, dy), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"scale {s:.2f}, shift ({dx},{dy})", fontsize=8.2); ax.axis("off")
    ax = fig.add_subplot(gs[0, 3])
    invariant = np.mean(np.stack(exemplars), axis=0)
    ax.imshow(invariant, cmap="magma"); ax.set_title("group-orbit average", fontsize=8.2); ax.axis("off")

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(probe_scales, direct, "o-", color=COLORS["gray"], lw=1.5, ms=3.5, label="single template")
    ax.plot(probe_scales, pooled, "o-", color=COLORS["blue"], lw=1.8, ms=3.5, label="max over transformed templates")
    ax.set(xlabel="probe scale", ylabel="normalized match", ylim=(-0.05, 1.05), title="Averaging/pooling removes nuisance sensitivity")
    ax.legend()

    ax = fig.add_subplot(gs[1, 2:])
    x = np.linspace(-1, 1, 300)
    target = 0.36
    traj = [0.92]
    eta = 0.34
    for _ in range(12): traj.append(traj[-1] - eta * (traj[-1] - target))
    ax.plot(x, 0.5 * (x-target)**2, color=COLORS["purple"], lw=2)
    ax.scatter(traj, 0.5 * (np.array(traj)-target)**2, c=np.arange(len(traj)), cmap="viridis", s=25, zorder=3)
    # Label moved up/left into the empty interior of the bowl. It previously sat on
    # the curve and ran into the right spine, and its leader lay along the curve.
    # The leader now targets an early, well-separated iterate on the descending arc.
    ax.annotate(
        "negative feedback",
        xy=(traj[2], 0.5 * (traj[2] - target) ** 2),
        xytext=(0.10, 0.46),
        ha="left", va="center", fontsize=8, color=COLORS["navy"],
        arrowprops=dict(
            arrowstyle="-|>", color=COLORS["gray"], lw=1.1,
            shrinkA=4, shrinkB=7, connectionstyle="arc3,rad=-0.22",
        ),
    )
    ax.set(xlabel="presentation parameter", ylabel="mismatch", title="Canonicalization drives input to a standard", yticks=[], ylim=(-0.045, 1.0))
    save_figure(fig, outdir, "fig04_universals_invariance")
    return {
        "number_of_transformed_templates": len(exemplars),
        "direct_match_mean": float(direct.mean()),
        "pooled_match_mean": float(pooled.mean()),
        "pooled_match_min": float(pooled.min()),
        "canonicalization_initial_error": float(abs(traj[0] - target)),
        "canonicalization_final_error": float(abs(traj[-1] - target)),
    }


def retinal_movie(n_space: int = 128, n_time: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1, 1, n_space)
    t = np.linspace(0, 1, n_time)
    movie = np.ones((n_time, n_space))
    # Global illumination varies over 30-fold while a dark compact object moves.
    illum = np.exp(np.linspace(math.log(0.25), math.log(7.5), n_time))
    center = -0.80 + 1.60 * t
    width = 0.10
    for k in range(n_time):
        disk = np.exp(-0.5 * ((x - center[k]) / width) ** 8)
        movie[k] = illum[k] * (1.0 - 0.76 * disk)
    # A brief global dimming step near the end.
    movie[t > 0.77] *= 0.42
    return x, t, movie


def retina_operations(movie: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-6
    local = gaussian_filter(movie, sigma=(0, 3.0))
    surround = gaussian_filter(movie, sigma=(0, 11.0))
    norm = (local - surround) / (np.abs(surround) + eps)
    contrast = np.abs(norm)
    darkness = np.maximum(-norm, 0)
    # Small-dark-object/convexity proxy: center darkness suppressed by broad occupancy.
    broad_dark = gaussian_filter(darkness, sigma=(0, 8.0))
    convexity = np.maximum(darkness - 0.55 * broad_dark, 0)
    moving_edge = np.abs(np.diff(norm, axis=0, prepend=norm[[0]]))
    log_mean = np.log(gaussian_filter(movie, sigma=(0, 18.0)) + eps)
    dimming = np.maximum(-np.diff(log_mean, axis=0, prepend=log_mean[[0]]), 0)
    # Convert fields into population readouts; smooth in time as a firing-rate proxy.
    # Read a representative receptive field centered in the displayed retina.
    # This avoids the uninformative population maximum that simply follows the
    # moving object and makes the temporal selectivity visible.
    xx = np.linspace(-1.0, 1.0, movie.shape[1])
    rf = np.exp(-0.5 * (xx / 0.14) ** 2); rf /= rf.sum()
    outputs = {
        "sustained contrast": gaussian_filter(contrast @ rf, sigma=2.0),
        "net convexity": gaussian_filter(convexity @ rf, sigma=2.0),
        "moving edge": gaussian_filter(moving_edge @ rf, sigma=1.4),
        "net dimming": gaussian_filter(dimming @ rf, sigma=1.4),
    }
    for key, val in outputs.items(): outputs[key] = val / max(float(val.max()), eps)
    return {"field_contrast": contrast, "field_convexity": convexity, "field_moving": moving_edge, "field_dimming": dimming, **outputs}


def figure_frog_channels(outdir: Path) -> dict:
    x, t, movie = retinal_movie()
    out = retina_operations(movie)
    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.85, 1.0, 0.75], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(np.log10(movie + 1e-6), aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]], cmap="gray", vmin=-0.7, vmax=0.9)
    ax.set(title="Stimulus: a moving dark compact object, a 30-fold illumination ramp, then global dimming", ylabel="time", xlabel="retinal position")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01); cb.set_label("log$_{10}$ intensity")
    fields = [("field_contrast", "sustained contrast"), ("field_convexity", "net convexity proxy"), ("field_moving", "moving edge"), ("field_dimming", "net dimming")]
    cmaps = ["Blues", "Purples", "Oranges", "Greens"]
    for j, ((key, title), cmap) in enumerate(zip(fields, cmaps)):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(out[key], aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]], cmap=cmap)
        ax.set_title(title, fontsize=8.7); ax.set_xlabel("position")
        if j == 0: ax.set_ylabel("time")
        else: ax.set_yticklabels([])
    ax = fig.add_subplot(gs[2, :])
    colors = [COLORS["blue"], COLORS["purple"], COLORS["orange"], COLORS["green"]]
    for (key, _), color in zip(fields, colors): ax.plot(t, out[key.replace("field_", "") if False else {"field_contrast":"sustained contrast", "field_convexity":"net convexity", "field_moving":"moving edge", "field_dimming":"net dimming"}[key]], color=color, lw=1.7, label={"field_contrast":"contrast", "field_convexity":"convexity", "field_moving":"moving edge", "field_dimming":"dimming"}[key])
    ax.axvline(0.77, color=COLORS["red"], ls="--", lw=1, label="global dimming")
    ax.set(xlabel="time", ylabel="normalized receptive-field response", ylim=(-0.03, 1.08))
    # Legend moved below the axes; at "upper center" it lay across the curve peaks
    # and the dimming transient.
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=7.6,
              handletextpad=0.6, columnspacing=2.0, borderpad=0.2)
    save_figure(fig, outdir, "fig05_frog_parallel_channels")

    idx_ramp = (t > 0.12) & (t < 0.70)
    idx_dim = t > 0.77
    return {
        "model_status": "qualitative normalized LN/LNP-style pedagogical model; not fitted to historical data",
        "illumination_ratio": float(movie[idx_ramp].max() / max(movie[idx_ramp].min(), 1e-9)),
        "channel_peak_times": {k: float(t[np.argmax(out[k])]) for k in ["sustained contrast", "net convexity", "moving edge", "net dimming"]},
        "dimming_selectivity_ratio": float(out["net dimming"][idx_dim].mean() / max(out["net dimming"][idx_ramp].mean(), 1e-9)),
    }


def lnp_spikes(rng: np.random.Generator, stimulus: np.ndarray, filt: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    n_lag = len(filt)
    drive = np.zeros_like(stimulus)
    windows = np.lib.stride_tricks.sliding_window_view(stimulus, n_lag)
    # A spike at time t depends on the preceding n_lag stimulus samples.
    drive[n_lag:] = windows[:-1] @ filt
    if kind == "linear": rate = np.exp(0.55 * drive - 2.2)
    elif kind == "energy": rate = np.exp(0.22 * drive**2 - 2.3)
    else: raise ValueError(kind)
    rate = np.minimum(rate, 0.45)
    spikes = rng.random(len(rate)) < rate
    return drive, spikes.astype(int)


def spike_triggered_average(stimulus: np.ndarray, spikes: np.ndarray, n_lag: int) -> np.ndarray:
    idx = np.flatnonzero(spikes)[np.flatnonzero(spikes) >= n_lag]
    if len(idx) == 0: return np.zeros(n_lag)
    return np.mean(np.stack([stimulus[i-n_lag:i] for i in idx]), axis=0)


def figure_system_identification(outdir: Path, rng: np.random.Generator) -> dict:
    n = 120_000; n_lag = 40
    stimulus = rng.normal(size=n)
    tau = np.arange(n_lag)
    filt = np.exp(-tau / 8.0) - 0.72 * np.exp(-tau / 18.0)
    filt -= filt.mean(); filt /= np.linalg.norm(filt)
    _, spikes_l = lnp_spikes(rng, stimulus, filt, "linear")
    _, spikes_e = lnp_spikes(rng, stimulus, filt, "energy")
    sta_l = spike_triggered_average(stimulus, spikes_l, n_lag); sta_e = spike_triggered_average(stimulus, spikes_e, n_lag)
    sta_l /= max(np.linalg.norm(sta_l), 1e-12); sta_e /= max(np.linalg.norm(sta_e), 1e-12)
    fplot = filt
    corr_l = abs(normalized_corr(fplot, sta_l)); corr_e = abs(normalized_corr(fplot, sta_e))
    # Spike-triggered covariance: leading deviation direction recovers an even-symmetric feature.
    idx = np.flatnonzero(spikes_e); idx = idx[idx >= n_lag][:10000]
    X = np.stack([stimulus[i-n_lag:i] for i in idx])
    C = np.cov(X, rowvar=False) - np.eye(n_lag)
    vals, vecs = np.linalg.eigh(C)
    stc = vecs[:, np.argmax(np.abs(vals))]
    corr_stc = abs(normalized_corr(fplot, stc))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    ax = axes[0]; ax.plot(tau, fplot, color=COLORS["navy"], lw=2, label="true filter"); ax.plot(tau, sta_l, color=COLORS["blue"], lw=1.6, label="STA")
    ax.set(title=f"A  Linear feature: STA succeeds ($|r|={corr_l:.2f}$)", xlabel="lag", ylabel="normalized amplitude"); ax.legend(fontsize=7.5)
    ax = axes[1]; ax.plot(tau, fplot, color=COLORS["navy"], lw=2, label="true filter"); ax.plot(tau, sta_e, color=COLORS["red"], lw=1.4, label="STA")
    ax.set(title=f"B  Contrast energy: STA cancels ($|r|={corr_e:.2f}$)", xlabel="lag"); ax.legend(fontsize=7.5)
    ax = axes[2]; ax.plot(tau, fplot, color=COLORS["navy"], lw=2, label="true filter"); ax.plot(tau, stc * np.sign(np.dot(stc, fplot)), color=COLORS["green"], lw=1.6, label="STC direction")
    ax.set(title=f"C  Second-order statistic recovers it ($|r|={corr_stc:.2f}$)", xlabel="lag"); ax.legend(fontsize=7.5)
    save_figure(fig, outdir, "fig06_adequate_stimulus_identification")
    return {"n_samples": n, "linear_spikes": int(spikes_l.sum()), "energy_spikes": int(spikes_e.sum()), "sta_filter_correlation_linear": corr_l, "sta_filter_correlation_energy": corr_e, "stc_filter_correlation_energy": corr_stc}


def figure_synthesis(outdir: Path) -> dict:
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4)
    boxes = [
        (0.3, 2.35, 2.25, 1.05, "Microscopic substrate", "spikes, synapses, delays", COLORS["gray"]),
        (3.05, 2.35, 2.25, 1.05, "Formal dynamics", r"$\mathbf{x}(t+1)=F(\mathbf{x}(t))$", COLORS["blue"]),
        (5.80, 2.35, 2.25, 1.05, "Invariant operation", "predicate / feature / orbit", COLORS["purple"]),
        (8.55, 2.35, 2.25, 1.05, "Behavioral relevance", "memory, choice, prey, escape", COLORS["orange"]),
    ]
    for x, y, w, h, title, sub, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.08", facecolor="white", edgecolor=color, lw=1.8)
        ax.add_patch(patch); ax.text(x+w/2, y+0.67, title, ha="center", va="center", weight="bold", color=color, fontsize=9)
        ax.text(x+w/2, y+0.32, sub, ha="center", va="center", color=COLORS["navy"], fontsize=8)
    for x in [2.55, 5.30, 8.05]: arrow(ax, (x+0.05, 2.88), (x+0.43, 2.88), COLORS["navy"])
    ax.text(5.5, 1.55, "McCulloch's recurring question", ha="center", color=COLORS["navy"], weight="bold", fontsize=10)
    ax.text(5.5, 1.09, "Which organization of matter makes a specified relation invariant and operational?", ha="center", color=COLORS["navy"], fontsize=10)
    ax.plot([1.4, 9.7], [0.63, 0.63], color=COLORS["light"], lw=5, solid_capstyle="round")
    ax.text(1.4, 0.34, "1943: synthesis", ha="center", fontsize=8, color=COLORS["blue"])
    ax.text(4.15, 0.34, "1945: topology", ha="center", fontsize=8, color=COLORS["red"])
    ax.text(6.9, 0.34, "1947: invariance", ha="center", fontsize=8, color=COLORS["purple"])
    ax.text(9.7, 0.34, "1959: experiment", ha="center", fontsize=8, color=COLORS["green"])
    for x, c in [(1.4, COLORS["blue"]), (4.15, COLORS["red"]), (6.9, COLORS["purple"]), (9.7, COLORS["green"])]: ax.scatter(x, 0.63, s=55, color=c, edgecolor="white", zorder=3)
    save_figure(fig, outdir, "fig07_research_program")
    return {"through_line": "physical substrate -> formal dynamics -> invariant operation -> behavior"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("numerical_results.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    rng = np.random.default_rng(args.seed)
    results = {
        "metadata": {
            "seed": args.seed,
            "purpose": "pedagogical replication and formal demonstration",
            "historical_data_digitized": False,
            "formats": ["pdf", "png", "svg"],
        },
        "fig01_historical_timeline": figure_timeline(args.output),
        "fig02_threshold_logic": figure_logic(args.output),
        "fig03_feedback_heterarchy": figure_dynamics_heterarchy(args.output),
        "fig04_universals_invariance": figure_universals(args.output),
        "fig05_frog_parallel_channels": figure_frog_channels(args.output),
        "fig06_adequate_stimulus_identification": figure_system_identification(args.output, rng),
        "fig07_research_program": figure_synthesis(args.output),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
