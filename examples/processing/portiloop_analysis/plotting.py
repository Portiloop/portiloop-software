"""Figures for event-locked responses and group comparisons."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .epochs import EventLockedResponses

BAND_LABELS = {
    "slow": "Slow-band EEG ({0:g} to {1:g} Hz)",
    "sigma": "Sigma envelope ({0:g} to {1:g} Hz)",
    "broadband": "Broadband EEG ({0:g} to {1:g} Hz)",
}


def plot_responses(responses: EventLockedResponses, bands: Sequence[str] = ("slow", "sigma"),
                   xlim: Optional[Sequence[float]] = None, title: Optional[str] = None):
    """Average response (black) and +/- 1 SEM (grey) of each band, aligned on the stimulation marker (red line).

    The baseline window, when there is one, is shaded.

    Returns:
        the matplotlib Figure.
    """
    settings = responses.settings
    fig, axes = plt.subplots(len(bands), 1, figsize=(10, 4.5 * len(bands)), sharex=True, layout="constrained",
                             squeeze=False)
    for ax, band in zip(axes[:, 0], bands):
        mean, sem = responses.mean(band), responses.sem(band)
        ax.plot(responses.time, mean, color="black", label="Mean")
        ax.fill_between(responses.time, mean - sem, mean + sem, color="gray", alpha=0.3, linewidth=0,
                        label="+/- 1 SEM")
        ax.axvline(0, color="red", linestyle="--", label="Stimulation")
        baseline = getattr(settings, f"{band}_baseline")
        if isinstance(baseline, tuple):
            ax.axvspan(*baseline, color="#0A9999", alpha=0.2, linewidth=0, label="Baseline")
        label = BAND_LABELS[band].format(*getattr(settings, f"{band}_band" if band != "broadband" else "broadband"))
        if baseline is not None:
            label = "Baseline-corrected\n" + label[0].lower() + label[1:]
        ax.set_ylabel(f"{label} (µV)")
        ax.legend(frameon=False, loc="upper right")
    axes[0, 0].set_title(title or f"{responses.channel}: {responses.n_kept}/{len(responses.events)} epochs kept")
    axes[-1, 0].set_xlabel("Time relative to the stimulation marker (s)")
    if xlim is not None:
        axes[-1, 0].set_xlim(xlim)
    return fig


def plot_groups(ax, table: pd.DataFrame, value: str, group: str, order: Optional[Sequence[str]] = None,
                colors: Optional[Sequence[str]] = None, ylabel: Optional[str] = None, seed: int = 0):
    """Box plot (whiskers at the minimum and maximum) with one dot per participant, for each group.

    Args:
        ax: matplotlib Axes to draw in.
        table: one row per participant.
        value: column with the values to plot.
        group: column with the group of each participant.
        order: groups in the order to plot them (default: order of appearance).
        colors: one color per group.
        ylabel: label of the y axis (default: the value column).
        seed: seed of the random horizontal jitter of the dots.
    """
    order = list(order) if order is not None else list(pd.unique(table[group]))
    colors = list(colors) if colors is not None else ["#F4A261", "#2166AC", "#2A9D8F", "#9B5DE5"][:len(order)]
    rng = np.random.default_rng(seed)
    for position, (name, color) in enumerate(zip(order, colors), start=1):
        values = table.loc[table[group] == name, value].dropna().to_numpy()
        if len(values) == 0:
            continue
        ax.boxplot(values, positions=[position], widths=0.35, whis=(0, 100), patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color, linewidth=2),
                   medianprops=dict(color=color, linewidth=2),
                   whiskerprops=dict(color=color, linewidth=2, linestyle="--"),
                   capprops=dict(color=color, linewidth=2))
        ax.scatter(position + rng.uniform(-0.08, 0.08, len(values)), values, s=70, color=color, alpha=0.6,
                   zorder=3, linewidths=0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, fontsize=12)
    ax.set_xlim(0.4, len(order) + 0.6)
    ax.set_ylabel(ylabel or value, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_correlation(ax, table: pd.DataFrame, x: str, y: str):
    """Scatter plot with a least-squares line and the Pearson correlation (r, p) in the title."""
    data = table[[x, y]].dropna()
    r, p = stats.pearsonr(data[x], data[y])
    fit = stats.linregress(data[x], data[y])
    ax.scatter(data[x], data[y], alpha=0.7)
    xs = np.linspace(data[x].min(), data[x].max(), 100)
    ax.plot(xs, fit.intercept + fit.slope * xs, color="red")
    ax.set_title(f"{x} vs {y}: r = {r:.3f}, p = {p:.3g}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax
