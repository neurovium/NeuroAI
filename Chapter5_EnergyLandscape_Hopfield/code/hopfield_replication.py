#!/usr/bin/env python3
"""Reproduce the numerical figures in the Hopfield student guide.

The experiments are deliberately compact and synthetic: they isolate the
mechanisms discussed in the text without requiring external data or a GPU.
Every run is deterministic for a fixed seed and writes figures as PDF, PNG,
and SVG together with a machine-readable JSON summary.

Usage
-----
python code/hopfield_replication.py --output figures --results numerical_results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf, logsumexp


COLORS = {
    "blue": "#2166ac",
    "red": "#b2182b",
    "gold": "#d99a2b",
    "green": "#1b7837",
    "purple": "#762a83",
    "gray": "#5b6573",
    "light": "#eef3f8",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def stacked_legend(
    ax: plt.Axes,
    entries: list[tuple[dict, str]],
    loc: str = "lower left",
    borderpad: float = 0.35,
    entry_sep: float = 4.0,
    fontsize: float | None = None,
) -> None:
    """Legend whose entries stack the handle above its label instead of beside it.

    Each entry is drawn as a short marked line with the text directly underneath,
    which keeps the block narrow enough to sit clear of vertical guide lines.
    The block is anchored inside the axes, so it cannot bleed past the spines.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, TextArea, VPacker

    fontsize = plt.rcParams["legend.fontsize"] if fontsize is None else fontsize
    handle_width = 20.0
    handle_height = 8.0
    blocks = []
    for style, label in entries:
        drawing = DrawingArea(handle_width, handle_height, 0.0, 0.0)
        line = Line2D(
            [1.0, handle_width / 2.0, handle_width - 1.0],
            [handle_height / 2.0] * 3,
            markevery=[1],
            **style,
        )
        drawing.add_artist(line)
        text = TextArea(label, textprops={"fontsize": fontsize})
        blocks.append(VPacker(children=[drawing, text], align="center", pad=0.0, sep=1.0))

    box = AnchoredOffsetbox(
        loc=loc,
        child=VPacker(children=blocks, align="left", pad=0.0, sep=entry_sep),
        frameon=False,
        borderpad=borderpad,
        pad=0.0,
    )
    ax.add_artist(box)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", facecolor="white")
    plt.close(fig)


def sign_nonzero(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1, -1).astype(np.int8)


def hopfield_weights(patterns: np.ndarray) -> np.ndarray:
    n = patterns.shape[1]
    weights = patterns.T.astype(float) @ patterns.astype(float) / n
    np.fill_diagonal(weights, 0.0)
    return weights


def hopfield_energy(state: np.ndarray, weights: np.ndarray) -> float:
    state_float = state.astype(float)
    return float(-0.5 * state_float @ weights @ state_float)


def asynchronous_recall(
    state: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    sweeps: int = 12,
    sample_every: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random sequential zero-temperature dynamics with efficient field updates."""
    state = state.copy().astype(np.int8)
    n = state.size
    field = weights @ state.astype(float)
    energies = [hopfield_energy(state, weights)]
    overlaps = [float(target.astype(float) @ state.astype(float)) / n]
    updates = [0]
    count = 0
    for _ in range(sweeps):
        for index in rng.permutation(n):
            proposed = 1 if field[index] >= 0.0 else -1
            if proposed != state[index]:
                delta = proposed - int(state[index])
                state[index] = proposed
                field += weights[:, index] * delta
            count += 1
            if count % sample_every == 0:
                energies.append(hopfield_energy(state, weights))
                overlaps.append(float(target.astype(float) @ state.astype(float)) / n)
                updates.append(count)
        if np.all(sign_nonzero(field) == state):
            break
    return state, np.asarray(updates), np.column_stack([energies, overlaps])


def make_timeline(output: Path) -> None:
    events = [
        (1943, "McCulloch--Pitts", "threshold units"),
        (1949, "Hebb", "activity-dependent coupling"),
        (1972, "Amari; Anderson; Kohonen", "associative dynamics"),
        (1974, "Little", "recurrent binary network"),
        (1982, "Hopfield, PNAS", "energy + content addressability"),
        (1984, "Hopfield, PNAS", "graded-response Lyapunov theory"),
        (1985, "Amit--Gutfreund--Sompolinsky", "spin-glass phase theory"),
        (1985, "Hopfield--Tank", "optimization by analog dynamics"),
        (2016, "Krotov--Hopfield", "dense associative memory"),
        (2017, "Demircigil et al.", "exponential capacity theorem"),
        (2021, "Modern Hopfield networks", "softmax update = attention case"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(0.5, color="#9aa6b2", lw=1.2)
    y = np.arange(len(events))[::-1]
    for k, ((year, title, phrase), yy) in enumerate(zip(events, y)):
        side = -1 if k % 2 == 0 else 1
        color = COLORS["blue"] if year < 1982 else (COLORS["red"] if year <= 1985 else COLORS["green"])
        ax.scatter([0.5], [yy], s=35, color=color, zorder=3, edgecolor="white", linewidth=0.6)
        x_text = 0.43 if side < 0 else 0.57
        ha = "right" if side < 0 else "left"
        ax.plot([0.5, x_text + (0.01 if side < 0 else -0.01)], [yy, yy], color=color, lw=0.8)
        ax.text(x_text, yy + 0.13, f"{year}  {title}", ha=ha, va="bottom", weight="bold", color=color)
        ax.text(x_text, yy - 0.10, phrase, ha=ha, va="top", color="#343b44")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.7, len(events) - 0.3)
    ax.axis("off")
    ax.set_title("A lineage, not a single-origin story", pad=10, weight="bold")
    fig.text(0.5, 0.01, "Blue: foundations   Red: classical Hopfield era   Green: later high-capacity formulations", ha="center", color=COLORS["gray"])
    save_figure(fig, output, "fig01_historical_timeline")


def classical_recall_demo(output: Path, rng: np.random.Generator) -> dict:
    n_side = 20
    n = n_side**2
    p = 18
    patterns = rng.choice([-1, 1], size=(p, n)).astype(np.int8)
    target = patterns[0]
    cue = target.copy()
    corrupted = rng.choice(n, size=int(0.28 * n), replace=False)
    cue[corrupted] *= -1
    weights = hopfield_weights(patterns)
    final, updates, trace = asynchronous_recall(cue, weights, target, rng, sweeps=10)
    energy = trace[:, 0]
    overlap = trace[:, 1]

    fig = plt.figure(figsize=(7.6, 4.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.38)
    top = outer[0].subgridspec(1, 3, wspace=0.45)
    # The lower-left panel carries a twin right-hand axis, so its outward-facing
    # red label needs a wider corridor than the image row above it.
    bottom = outer[1].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.62)
    labels = ["stored target", "28% corrupted cue", "retrieved state"]
    for col, (array, label) in enumerate(zip((target, cue, final), labels)):
        ax = fig.add_subplot(top[0, col])
        ax.imshow(array.reshape(n_side, n_side), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#7f8a96")
            spine.set_linewidth(0.6)

    ax_e = fig.add_subplot(bottom[0, 0])
    ax_e.plot(updates / n, energy / n, color=COLORS["blue"], lw=1.8)
    ax_e.set_xlabel("asynchronous updates (sweeps)")
    ax_e.set_ylabel(r"energy per neuron $E/N$", color=COLORS["blue"])
    ax_e.tick_params(axis="y", colors=COLORS["blue"])
    ax_e.grid(alpha=0.18)
    ax_m = ax_e.twinx()
    ax_m.plot(updates / n, overlap, color=COLORS["red"], lw=1.8)
    ax_m.set_ylabel(r"target overlap $m^1$", color=COLORS["red"])
    ax_m.tick_params(axis="y", colors=COLORS["red"])
    ax_m.spines["right"].set_visible(True)

    ax_d = fig.add_subplot(bottom[0, 1])
    delta_e = np.diff(energy)
    ax_d.hist(delta_e, bins=18, color=COLORS["gray"], edgecolor="white")
    ax_d.axvline(0, color=COLORS["red"], ls="--", lw=1)
    ax_d.set_xlabel(r"sampled $\Delta E$")
    ax_d.set_ylabel("count")
    ax_d.set_title("Lyapunov check")
    fig.suptitle("Classical Hopfield recall: a basin is a dynamical object", weight="bold", y=0.995)
    save_figure(fig, output, "fig02_classical_recall")
    return {
        "N": n,
        "patterns": p,
        "initial_overlap": float(target.astype(float) @ cue.astype(float)) / n,
        "final_overlap": float(target.astype(float) @ final.astype(float)) / n,
        "initial_energy_per_neuron": float(energy[0] / n),
        "final_energy_per_neuron": float(energy[-1] / n),
        "largest_sampled_energy_increase": float(np.max(delta_e)) if delta_e.size else 0.0,
    }


def t0_mean_field_branch(alphas: np.ndarray) -> np.ndarray:
    """Replica-symmetric T=0 retrieval branch by continuation."""
    branch = np.full((alphas.size, 3), np.nan)
    guess = np.array([0.999, 0.01, 1.02])  # m, C, r
    for idx, alpha in enumerate(alphas):
        def residual(x: np.ndarray) -> np.ndarray:
            m, c, r = x
            denom = max(alpha * r, 1e-12)
            return np.array(
                [
                    m - erf(m / math.sqrt(2.0 * denom)),
                    c - math.sqrt(2.0 / (math.pi * denom)) * math.exp(-(m * m) / (2.0 * denom)),
                    r - 1.0 / ((1.0 - c) ** 2),
                ]
            )

        result = least_squares(
            residual,
            guess,
            bounds=([1e-5, 0.0, 1.0], [1.0, 0.999, 1e4]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=4000,
        )
        if result.cost < 1e-12 and result.x[0] > 0.05:
            branch[idx] = result.x
            guess = result.x
        else:
            break
    return branch


def synchronous_low_rank_recall(patterns: np.ndarray, cue: np.ndarray, steps: int = 35) -> np.ndarray:
    state = cue.copy().astype(np.int8)
    p, n = patterns.shape
    for _ in range(steps):
        overlaps = patterns @ state.astype(float)
        field = patterns.T @ overlaps / n - (p / n) * state
        updated = sign_nonzero(field)
        if np.array_equal(updated, state):
            break
        state = updated
    return state


def capacity_experiment(output: Path, rng: np.random.Generator, quick: bool) -> dict:
    n = 500 if quick else 800
    alphas_sim = np.linspace(0.02, 0.24, 12)
    repetitions = 5 if quick else 10
    mean_overlap = []
    success = []
    for alpha in alphas_sim:
        p = max(1, int(round(alpha * n)))
        overlaps = []
        for _ in range(repetitions):
            patterns = rng.choice([-1, 1], size=(p, n)).astype(np.int8)
            cue = patterns[0].copy()
            flip = rng.choice(n, size=int(0.10 * n), replace=False)
            cue[flip] *= -1
            final = synchronous_low_rank_recall(patterns, cue)
            overlaps.append(float(patterns[0].astype(float) @ final.astype(float)) / n)
        mean_overlap.append(float(np.mean(overlaps)))
        success.append(float(np.mean(np.asarray(overlaps) > 0.90)))

    alphas_mf = np.linspace(0.001, 0.17, 340)
    branch = t0_mean_field_branch(alphas_mf)
    valid = np.isfinite(branch[:, 0])
    terminal_alpha = float(alphas_mf[np.where(valid)[0][-1]])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.30})
    ax = axes[0]
    ax.plot(alphas_mf[valid], branch[valid, 0], color=COLORS["red"], lw=2.0, label=r"RS $T=0$ retrieval branch")
    ax.axvline(0.138, color="#222222", ls="--", lw=1.1, label=r"$\alpha_c\simeq0.138$")
    ax.set_xlabel(r"load $\alpha=p/N$")
    ax.set_ylabel(r"retrieval overlap $m$")
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.18)
    ax.set_title("thermodynamic continuation")

    ax = axes[1]
    ax.plot(alphas_sim, mean_overlap, "o-", color=COLORS["blue"], label="mean final overlap")
    ax.plot(alphas_sim, success, "s-", color=COLORS["green"], label=r"fraction with $m>0.9$")
    ax.axvline(0.138, color="#222222", ls="--", lw=1.1)
    ax.set_xlabel(r"load $\alpha=p/N$")
    ax.set_ylabel("finite-network retrieval")
    ax.set_ylim(-0.03, 1.04)
    stacked_legend(
        ax,
        [
            (
                {"color": COLORS["blue"], "marker": "o", "markersize": 5, "lw": 1.5},
                "mean final overlap",
            ),
            (
                {"color": COLORS["green"], "marker": "s", "markersize": 5, "lw": 1.5},
                r"fraction with $m>0.9$",
            ),
        ],
        loc="lower left",
        fontsize=7.4,
        borderpad=0.7,
    )
    ax.grid(alpha=0.18)
    ax.set_title(fr"simulation: $N={n}$, 10% cue noise")
    fig.suptitle("Capacity is a criterion-dependent phase boundary", weight="bold", y=1.02)
    save_figure(fig, output, "fig03_capacity_transition")
    return {
        "simulation_N": n,
        "repetitions_per_load": repetitions,
        "alphas": alphas_sim.tolist(),
        "mean_final_overlap": mean_overlap,
        "retrieval_success_fraction": success,
        "mean_field_numerical_terminal_alpha": terminal_alpha,
        "reference_spinodal_alpha": 0.138,
    }


def inverse_tanh_integral(v: np.ndarray, gain: float) -> np.ndarray:
    clipped = np.clip(v, -1 + 1e-12, 1 - 1e-12)
    return (clipped * np.arctanh(clipped) + 0.5 * np.log(1.0 - clipped**2)) / gain


def graded_energy(v: np.ndarray, weights: np.ndarray, gain: float) -> float:
    return float(-0.5 * v @ weights @ v + np.sum(inverse_tanh_integral(v, gain)))


def graded_response_experiment(output: Path, rng: np.random.Generator) -> dict:
    n = 120
    p = 4
    gain = 2.2
    dt = 0.025
    steps = 2400
    patterns = rng.choice([-1, 1], size=(p, n)).astype(float)
    weights = hopfield_weights(patterns)
    cue = patterns[0].copy()
    cue[rng.choice(n, size=int(0.22 * n), replace=False)] *= -1
    v0 = 0.72 * cue
    u = np.arctanh(v0) / gain
    times = []
    energies = []
    overlaps = []
    for step in range(steps + 1):
        v = np.tanh(gain * u)
        if step % 8 == 0:
            times.append(step * dt)
            energies.append(graded_energy(v, weights, gain))
            overlaps.append(float(patterns[0] @ v) / n)
        u += dt * (-u + weights @ v)

    times = np.asarray(times)
    energies = np.asarray(energies)
    overlaps = np.asarray(overlaps)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.34})
    axes[0].plot(times, energies / n, color=COLORS["blue"], lw=1.8)
    axes[0].set_xlabel(r"time $t/\tau$")
    axes[0].set_ylabel(r"graded energy $E/N$")
    axes[0].set_title("Lyapunov descent")
    axes[0].grid(alpha=0.18)
    axes[1].plot(times, overlaps, color=COLORS["red"], lw=1.8)
    axes[1].set_xlabel(r"time $t/\tau$")
    axes[1].set_ylabel(r"continuous overlap $m^1$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("graded recall")
    axes[1].grid(alpha=0.18)
    axes[2].scatter(np.arange(40), np.tanh(gain * u)[:40], c=patterns[0, :40], cmap="RdBu_r", vmin=-1, vmax=1, s=19)
    axes[2].axhline(0, color="#777777", lw=0.7)
    axes[2].set_xlabel("neuron index (subset)")
    axes[2].set_ylabel(r"final output $V_i$")
    axes[2].set_title("continuous states")
    fig.suptitle("The 1984 graded-response model retains an energy function", weight="bold", y=1.04)
    save_figure(fig, output, "fig04_graded_response")
    return {
        "N": n,
        "patterns": p,
        "gain": gain,
        "initial_overlap": float(overlaps[0]),
        "final_overlap": float(overlaps[-1]),
        "initial_energy_per_neuron": float(energies[0] / n),
        "final_energy_per_neuron": float(energies[-1] / n),
        "largest_sampled_energy_increase": float(np.max(np.diff(energies))),
    }


def odd_double_factorial(k: int) -> int:
    if k <= 0:
        return 1
    out = 1
    for value in range(k, 0, -2):
        out *= value
    return out


def predicted_no_error_capacity(n_neurons: int, power: int) -> float:
    coefficient = 2.0 * odd_double_factorial(2 * power - 3)
    return n_neurons ** (power - 1) / (coefficient * math.log(n_neurons))


def dense_pattern_stable(patterns: np.ndarray, target_index: int, power: int) -> bool:
    """Check all bits using the exact energy-difference update of Krotov--Hopfield."""
    target = patterns[target_index].astype(np.int64)
    patterns64 = patterns.astype(np.int64)
    total_overlap = patterns64 @ target
    for i in range(target.size):
        base = total_overlap - patterns64[:, i] * target[i]
        score_plus = np.sum((base + patterns64[:, i]) ** power, dtype=np.int64)
        score_minus = np.sum((base - patterns64[:, i]) ** power, dtype=np.int64)
        proposed = 1 if score_plus >= score_minus else -1
        if proposed != target[i]:
            return False
    return True


def dense_capacity_experiment(output: Path, rng: np.random.Generator, quick: bool) -> dict:
    sizes = np.array([30, 45, 65, 90] if quick else [30, 45, 65, 90, 120])
    powers = [2, 3, 4]
    repetitions = 8 if quick else 16
    empirical: dict[int, list[float]] = {power: [] for power in powers}
    predicted: dict[int, list[int]] = {power: [] for power in powers}
    for power in powers:
        for n in sizes:
            k = max(2, int(round(predicted_no_error_capacity(int(n), power))))
            predicted[power].append(k)
            stable = 0
            for _ in range(repetitions):
                patterns = rng.choice([-1, 1], size=(k, int(n))).astype(np.int8)
                stable += int(dense_pattern_stable(patterns, 0, power))
            empirical[power].append(stable / repetitions)

    n_line = np.logspace(math.log10(20), math.log10(300), 180)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.30})
    color_for = {2: COLORS["blue"], 3: COLORS["red"], 4: COLORS["green"]}
    for power in powers:
        coefficient = 2.0 * odd_double_factorial(2 * power - 3)
        k_line = n_line ** (power - 1) / (coefficient * np.log(n_line))
        axes[0].loglog(n_line, k_line, color=color_for[power], lw=2, label=fr"$n={power}$")
        axes[0].scatter(sizes, predicted[power], color=color_for[power], s=20)
        axes[1].plot(sizes, empirical[power], "o-", color=color_for[power], label=fr"$n={power}$")
    axes[0].set_xlabel("neurons $N$")
    axes[0].set_ylabel(r"no-error scale $K_{\max}$")
    axes[0].set_title(r"$K_{\max}\sim N^{n-1}/\ln N$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18, which="both")
    axes[1].set_xlabel("neurons $N$")
    axes[1].set_ylabel("probability target is fully stable")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("exact energy-difference test")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.18)
    fig.suptitle("Polynomial dense associative memory changes the capacity exponent", weight="bold", y=1.02)
    save_figure(fig, output, "fig05_dense_capacity")
    return {
        "sizes": sizes.tolist(),
        "repetitions": repetitions,
        "predicted_no_error_K": {str(k): v for k, v in predicted.items()},
        "probability_all_bits_stable": {str(k): v for k, v in empirical.items()},
    }


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def modern_update(patterns: np.ndarray, query: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    weights = softmax(beta * (patterns @ query))
    return patterns.T @ weights, weights


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))


def modern_hopfield_experiment(output: Path, rng: np.random.Generator) -> dict:
    dimension = 64
    counts = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    betas = np.array([0.03, 0.06, 0.125, 0.25])
    trials = 24
    similarity = np.zeros((betas.size, counts.size))
    initial_similarity = np.zeros(counts.size)
    for j, count in enumerate(counts):
        for _ in range(trials):
            patterns = rng.normal(size=(int(count), dimension))
            patterns *= math.sqrt(dimension) / np.linalg.norm(patterns, axis=1, keepdims=True)
            target = patterns[0]
            query = target + rng.normal(scale=0.80, size=dimension)
            initial_similarity[j] += cosine(query, target) / trials
            for i, beta in enumerate(betas):
                updated, _ = modern_update(patterns, query, float(beta))
                similarity[i, j] += cosine(updated, target) / trials

    count = 64
    patterns = rng.normal(size=(count, dimension))
    patterns *= math.sqrt(dimension) / np.linalg.norm(patterns, axis=1, keepdims=True)
    target = patterns[0]
    query = target + rng.normal(scale=0.80, size=dimension)
    beta = 0.125
    state = query.copy()
    energies = []
    similarities = []
    final_weights = None
    for _ in range(9):
        energies.append(float(0.5 * state @ state - logsumexp(beta * (patterns @ state)) / beta))
        similarities.append(cosine(state, target))
        state, final_weights = modern_update(patterns, state, beta)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), constrained_layout=True)
    image = axes[0].imshow(similarity, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    axes[0].set_xticks(np.arange(counts.size), labels=counts, rotation=45)
    axes[0].set_yticks(np.arange(betas.size), labels=[f"{x:.3g}" for x in betas])
    axes[0].set_xlabel("stored patterns $K$")
    axes[0].set_ylabel(r"inverse temperature $\beta$")
    axes[0].set_title("post-update cosine")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)

    order = np.argsort(final_weights)[::-1]
    axes[1].bar(np.arange(12), final_weights[order[:12]], color=COLORS["blue"])
    axes[1].set_xticks(np.arange(12), labels=[str(x) for x in order[:12]], rotation=70)
    axes[1].set_xlabel("memory index (top weights)")
    axes[1].set_ylabel("softmax weight")
    axes[1].set_title("attention concentrates")

    axes[2].plot(np.arange(len(energies)), energies, "o-", color=COLORS["blue"], label="energy")
    axes[2].set_xlabel("fixed-point iteration")
    axes[2].set_ylabel("modern Hopfield energy", color=COLORS["blue"])
    axes[2].tick_params(axis="y", colors=COLORS["blue"])
    ax_sim = axes[2].twinx()
    ax_sim.plot(np.arange(len(similarities)), similarities, "s-", color=COLORS["red"], label="cosine")
    ax_sim.set_ylabel("target cosine", color=COLORS["red"])
    ax_sim.tick_params(axis="y", colors=COLORS["red"])
    ax_sim.spines["right"].set_visible(True)
    axes[2].set_title("fixed-point descent")
    fig.suptitle("Modern continuous Hopfield retrieval is a softmax-weighted memory lookup", weight="bold", y=1.04)
    save_figure(fig, output, "fig06_modern_hopfield")
    return {
        "dimension": dimension,
        "stored_pattern_counts": counts.tolist(),
        "betas": betas.tolist(),
        "trials": trials,
        "mean_initial_cosine": initial_similarity.tolist(),
        "mean_post_update_cosine": similarity.tolist(),
        "example_energy": energies,
        "example_target_cosine": similarities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("numerical_results.json"))
    parser.add_argument("--seed", type=int, default=1982)
    parser.add_argument("--quick", action="store_true", help="Use fewer Monte Carlo repetitions.")
    args = parser.parse_args()

    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    make_timeline(args.output)
    results = {
        "seed": args.seed,
        "quick": args.quick,
        "classical_recall": classical_recall_demo(args.output, rng),
        "capacity": capacity_experiment(args.output, rng, args.quick),
        "graded_response": graded_response_experiment(args.output, rng),
        "dense_capacity": dense_capacity_experiment(args.output, rng, args.quick),
        "modern_hopfield": modern_hopfield_experiment(args.output, rng),
    }
    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
