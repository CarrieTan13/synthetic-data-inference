"""Shared forest-plot rendering for the task-exchangeability experiments.

Visual style (chosen: "Variant 1")
-----------------------------------
* White background, no shaded band, no dashed reference line.
* A *primary* interval per row (Algorithm 1, Algorithm 2, the one-piece
  conformal, or the naive synth-only baseline when it is the focus) drawn as a
  thick error bar, **coloured by coverage**: green if it covers the truth, red
  if it misses. The colour pair is configurable (e.g. blue/red for the
  one-piece comparison).
* The ground-truth theta* is a black 'x' sitting *on the primary interval's
  line* (per request).
* An optional *baseline* interval (the naive synth-only CI) is drawn as a thin
  bar a small offset below the primary, in a fixed muted colour.
* Rows are packed tightly; figures are wide and short (~1:1) for the
  small/medium task sets, and naturally tall for the all-task AutoRater panels.

Public API
----------
Record                          -- one task row (label, theta*, named intervals)
SYN_ONLY/ALG1/ALG2/ONEPIECE/NAIVE  -- canonical method keys + labels
COV_GREEN_RED / COV_BLUE_RED    -- coverage colour pairs
select_records(...)             -- sort / subsample tasks
forest_panel(ax, ...)           -- draw one forest panel onto an Axes
single_panel(...)               -- one-panel figure -> png (+pdf)
two_panel(...)                  -- two-panel figure -> png (+pdf)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Method keys / labels and colour conventions.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Method:
    key: str
    label: str


SYN_ONLY = Method("syn_only", "naive synth-only")
NAIVE = SYN_ONLY                       # alias
ALG1 = Method("alg1", "Algorithm 1")
ALG2 = Method("alg2", "Algorithm 2")
ALG4 = Method("alg4", "Algorithm 4 (finite-sample)")
ONEPIECE = Method("onepiece", "Empirical-Gap Conformal (one-piece)")

# Coverage is intentionally NOT encoded by colour: every primary CI is drawn in
# the same blue, whether or not it covers theta*. (The pair names are kept so
# existing imports/callers keep working; both entries are now blue.)
_BLUE = "#1f77b4"
COV_GREEN_RED = (_BLUE, _BLUE)
COV_BLUE_RED = (_BLUE, _BLUE)

BASELINE_COLOR = "#EE7733"             # naive synth-only when shown as baseline
TRUTH_COLOR = "#000000"
TRUTH_MARKER = "x"

# --------------------------------------------------------------------------- #
# Newer per-task interval style: seaborn "Set2" palette drawn as a lightened
# bar with a darker outline; task exchangeability in blue, the naive
# synthetic-only baseline in orange, and the ground-truth theta* as a thin
# gray cross.
# --------------------------------------------------------------------------- #

_SET2 = {
    "green": (0.4, 0.7607843137254902, 0.6470588235294118),
    "orange": (0.9882352941176471, 0.5529411764705882, 0.38431372549019605),
    "blue": (0.5529411764705882, 0.6274509803921569, 0.796078431372549),
}
_TRUTH_GRAY = "0.4"

# friendlier legend labels for the canonical methods
_LEGEND_LABEL = {SYN_ONLY.key: "naive synthetic-only",
                 ALG1.key: "task exchangeability"}


def _llabel(method: "Method") -> str:
    return _LEGEND_LABEL.get(method.key, method.label)


def _lighten(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc

    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def _bar(ax, x0, x1, y, base, fill_lw, stroke_lw):
    """Draw a horizontal interval as a lightened fill with a darker outline."""
    import matplotlib.patheffects as pe

    ax.plot(
        [x0, x1], [y, y], linewidth=fill_lw, color=_lighten(base, 0.6),
        solid_capstyle="butt", zorder=3,
        path_effects=[
            pe.Stroke(linewidth=stroke_lw, offset=(-1, 0), foreground=base),
            pe.Stroke(linewidth=stroke_lw, offset=(1, 0), foreground=base),
            pe.Normal(),
        ],
    )


@dataclass
class Record:
    label: str
    theta: float
    intervals: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)

    def add(self, method: Method, lo: float, hi: float, covered: Optional[bool] = None) -> "Record":
        if covered is None:
            covered = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= self.theta <= hi)
        self.intervals[method.key] = (float(lo), float(hi), bool(covered))
        return self


# --------------------------------------------------------------------------- #
# Task selection.
# --------------------------------------------------------------------------- #


def select_records(
    records: Sequence[Record],
    *,
    n_show: Optional[int] = None,
    mode: str = "all",
    seed: int = 0,
) -> List[Record]:
    """Sort by theta* and optionally subsample to n_show rows."""
    recs = sorted(records, key=lambda r: r.theta)
    if mode == "all" or n_show is None or n_show >= len(recs):
        return recs
    if mode == "head":
        chosen = recs[:n_show]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(recs), size=n_show, replace=False))
        chosen = [recs[i] for i in idx]
    else:  # "spread": evenly spaced across the theta* range
        idx = np.unique(np.linspace(0, len(recs) - 1, n_show).round().astype(int))
        chosen = [recs[i] for i in idx]
    return sorted(chosen, key=lambda r: r.theta)


# --------------------------------------------------------------------------- #
# Core panel.
# --------------------------------------------------------------------------- #


def _clip(v: float, clip: Optional[Tuple[float, float]]) -> float:
    if clip is None or not np.isfinite(v):
        return v
    return min(clip[1], max(clip[0], v))


def _expand(lo: float, hi: float, min_w: float,
            clip: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    """Widen a too-narrow interval (for rendering only) so it stays visible."""
    if min_w and np.isfinite(lo) and np.isfinite(hi) and (hi - lo) < min_w:
        c = 0.5 * (lo + hi)
        lo, hi = c - 0.5 * min_w, c + 0.5 * min_w
        if clip is not None:
            lo, hi = max(clip[0], lo), min(clip[1], hi)
    return lo, hi


def forest_panel(
    ax,
    records: Sequence[Record],
    *,
    primary: Method,
    baseline: Optional[Method] = None,
    cov_colors: Tuple[str, str] = COV_GREEN_RED,
    clip: Optional[Tuple[float, float]] = None,
    thick: bool = False,
    sort: bool = True,
    show_ylabels: bool = True,
    title: Optional[str] = None,
    xlabel: str = r"$\theta$",
    xlim: Optional[Tuple[float, float]] = None,
    primary_lw: Optional[float] = None,
    baseline_offset: float = 0.18,
    ylabel_fontsize: float = 10.0,
    xlabel_fontsize: float = 16.0,
    title_fontsize: float = 14.0,
    truth_ms: Optional[float] = None,
    truth_mew: Optional[float] = None,
    baseline_lw: float = 1.4,
    baseline_ms: float = 2.6,
    baseline_capsize: float = 1.8,
    autoscale_x: bool = False,
    min_render_width: float = 0.0,
    baseline_fill_lw: Optional[float] = None,
    right_text: Optional[Dict[str, str]] = None,
    right_header: Optional[str] = None,
    right_fontsize: float = 8.0,
    truth_ci: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Draw one forest panel; return a small summary dict (coverage, width)."""
    recs = sorted(records, key=lambda r: r.theta) if sort else list(records)
    n = len(recs)

    fill_lw = primary_lw if primary_lw is not None else (10.0 if thick else 5.0)
    stroke_lw = fill_lw + 2.0
    t_ms = truth_ms if truth_ms is not None else (11 if thick else 9)
    t_mew = truth_mew if truth_mew is not None else (2.4 if thick else 1.8)
    off = 0.5 * baseline_offset if baseline is not None else 0.0
    b_fill = baseline_fill_lw if baseline_fill_lw is not None else fill_lw
    b_stroke = b_fill + 2.0

    p_cov: List[bool] = []
    p_w: List[float] = []
    p_w_clip: List[float] = []
    b_cov: List[bool] = []

    for i, r in enumerate(recs):
        lo, hi, cov = r.intervals[primary.key]
        p_cov.append(bool(cov))
        if np.isfinite(lo) and np.isfinite(hi):
            p_w.append(hi - lo)
            if clip is not None:
                p_w_clip.append(_clip(hi, clip) - _clip(lo, clip))
        dlo, dhi = _expand(_clip(lo, clip), _clip(hi, clip), min_render_width, clip)

        # baseline (naive synth-only) bar, just below the primary
        if baseline is not None and baseline.key in r.intervals:
            blo, bhi, bc = r.intervals[baseline.key]
            b_cov.append(bool(bc))
            bdlo, bdhi = _expand(_clip(blo, clip), _clip(bhi, clip),
                                 min_render_width, clip)
            _bar(ax, bdlo, bdhi, i - off, _SET2["orange"], b_fill, b_stroke)

        # primary (task-exchangeability) bar
        _bar(ax, dlo, dhi, i + off, _SET2["blue"], fill_lw, stroke_lw)

        # ground-truth theta*: a black gold-rating CI bar when available, else a cross
        tci = truth_ci.get(r.label) if truth_ci else None
        if tci is not None and np.isfinite(tci[0]) and np.isfinite(tci[1]):
            tlo, thi = _clip(tci[0], clip), _clip(tci[1], clip)
            cap = 0.22
            ax.plot([tlo, thi], [i, i], color="black", lw=1.1, zorder=7,
                    solid_capstyle="butt")
            ax.plot([tlo, tlo], [i - cap, i + cap], color="black", lw=1.1, zorder=7)
            ax.plot([thi, thi], [i - cap, i + cap], color="black", lw=1.1, zorder=7)
            ax.plot(r.theta, i, "|", color="black", ms=6, mew=1.1, zorder=8)
        else:
            ax.plot(r.theta, i, "x", color=_TRUTH_GRAY, ms=t_ms, mew=t_mew, zorder=6)

    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels([r.label for r in recs], fontsize=ylabel_fontsize)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=max(9, xlabel_fontsize - 2))
    ax.set_ylim(-0.7, n - 0.3)            # tight vertical packing
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif autoscale_x:
        vals = []
        for r in recs:
            for k in (primary.key, baseline.key if baseline is not None else None):
                if k and k in r.intervals:
                    lo, hi, _ = r.intervals[k]
                    if np.isfinite(lo): vals.append(_clip(lo, clip))
                    if np.isfinite(hi): vals.append(_clip(hi, clip))
            vals.append(r.theta)
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.04 * (hi - lo + 1e-9)
            ax.set_xlim(lo - pad, hi + pad)
    elif clip is not None:
        pad = 0.02 * (clip[1] - clip[0])
        ax.set_xlim(clip[0] - pad, clip[1] + pad)
    ax.spines[["top", "right"]].set_visible(False)

    # optional right-margin annotations (one monospace line per row, keyed by label)
    if right_text:
        x0 = 1.015
        ytrans = ax.get_yaxis_transform()
        for i, r in enumerate(recs):
            s = right_text.get(r.label)
            if s:
                ax.text(x0, i, s, transform=ytrans, va="center", ha="left",
                        family="monospace", fontsize=right_fontsize, clip_on=False)
        if right_header:
            # just above the top row, below the (padded) title, to avoid overlap
            ax.text(x0, n - 0.35, right_header, transform=ytrans, va="bottom",
                    ha="left", family="monospace", fontsize=right_fontsize,
                    fontweight="bold", clip_on=False)

    cov_pct = 100.0 * np.mean(p_cov) if p_cov else float("nan")
    med_w = np.median([w for w in p_w if np.isfinite(w)]) if p_w else float("nan")
    med_w_clip = (np.median([w for w in p_w_clip if np.isfinite(w)])
                  if p_w_clip else float("nan"))
    summ = dict(coverage=cov_pct, median_width=med_w, median_width_clipped=med_w_clip,
                n=n, baseline_coverage=100.0 * np.mean(b_cov) if b_cov else float("nan"))
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=16.0 if right_text else 6.0)
    return summ


# --------------------------------------------------------------------------- #
# Figure helpers (legend, sizing, save).
# --------------------------------------------------------------------------- #


def _legend_handles(primary: Method, baseline: Optional[Method],
                    cov_colors: Tuple[str, str], thick: bool,
                    truth_as_ci: bool = False):
    from matplotlib.lines import Line2D
    # Set2 bar swatches: orange baseline, blue primary, gray cross for truth.
    h = []
    if baseline is not None:
        h.append(Line2D([0], [0], color=_lighten(_SET2["orange"], 0.6), lw=8,
                        label=_llabel(baseline)))
    h.append(Line2D([0], [0], color=_lighten(_SET2["blue"], 0.6), lw=8,
                    label=_llabel(primary)))
    if truth_as_ci:
        h.append(Line2D([0], [0], color="black", lw=1.1, marker="|", mew=1.1, ms=6,
                        label=r"true $\theta^{\ast}$ (human-rating CI)"))
    else:
        h.append(Line2D([0], [0], marker="x", ls="none", color=_TRUTH_GRAY,
                        mew=1.8, ms=9, label=r"true $\theta^{\ast}$"))
    return h


#: Shared row density (inches per task row) + fixed margin, so every dataset's
#: forest plot has the same vertical spacing between rows and an adaptive height.
ROW_INCHES = 0.30
BASE_INCHES = 1.5


def _fig_height(n: int) -> float:
    """Adaptive height with a consistent per-row spacing across all datasets."""
    return max(3.0, ROW_INCHES * n + BASE_INCHES)


def _save(fig, out_path: str, also_pdf: bool = True):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if also_pdf and out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4] + ".pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"saved {out_path}" + (" (+ .pdf)" if also_pdf and out_path.endswith('.png') else ""))


def single_panel(
    records: Sequence[Record],
    *,
    primary: Method,
    baseline: Optional[Method],
    alpha: float,
    out_path: str,
    cov_colors: Tuple[str, str] = COV_GREEN_RED,
    clip: Optional[Tuple[float, float]] = None,
    thick: bool = False,
    title: Optional[str] = None,
    suptitle: Optional[str] = None,
    xlabel: str = r"$\theta$",
    xlim: Optional[Tuple[float, float]] = None,
    show_ylabels: bool = True,
    fig_width: float = 8.4,
    fig_height: Optional[float] = None,
    legend_fontsize: float = 11.0,
    **panel_kw,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted(records, key=lambda r: r.theta)
    h = fig_height if fig_height is not None else _fig_height(len(recs))
    fig, ax = plt.subplots(figsize=(fig_width, h))
    forest_panel(ax, recs, primary=primary, baseline=baseline, cov_colors=cov_colors,
                 clip=clip, thick=thick, sort=False, show_ylabels=show_ylabels,
                 title=title, xlabel=xlabel, xlim=xlim, **panel_kw)
    ax.legend(handles=_legend_handles(primary, baseline, cov_colors, thick,
                                      truth_as_ci=bool(panel_kw.get("truth_ci"))),
              loc="lower right", fontsize=legend_fontsize, framealpha=0.95,
              borderpad=0.8, handlelength=1.6, handletextpad=0.7, borderaxespad=0.6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


def two_panel(
    records: Sequence[Record],
    *,
    left_primary: Method, right_primary: Method,
    alpha: float,
    out_path: str,
    left_baseline: Optional[Method] = None, right_baseline: Optional[Method] = None,
    cov_colors: Tuple[str, str] = COV_GREEN_RED,
    left_cov_colors: Optional[Tuple[str, str]] = None,
    right_cov_colors: Optional[Tuple[str, str]] = None,
    clip: Optional[Tuple[float, float]] = None,
    thick: bool = False,
    left_title: Optional[str] = "", right_title: Optional[str] = "",
    suptitle: Optional[str] = None,
    xlabel: str = r"$\theta$",
    left_xlabel: Optional[str] = None, right_xlabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    autoscale_x: bool = False,
    share_x: bool = True,
    fig_width: float = 13.0,
    fig_height: Optional[float] = None,
    legend: bool = True,
) -> str:
    """Two side-by-side forest panels sharing the same row order (sorted by theta*).

    Each panel may use its own coverage colour pair (left_cov_colors /
    right_cov_colors); both default to cov_colors.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lcc = left_cov_colors or cov_colors
    rcc = right_cov_colors or cov_colors

    recs = sorted(records, key=lambda r: r.theta)
    h = fig_height if fig_height is not None else _fig_height(len(recs))
    # NB: not sharey -- a shared y-axis clears the left panel's tick labels.
    # Both panels get identical ylim from forest_panel, so rows still align.
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, h), sharey=False)
    sL = forest_panel(axes[0], recs, primary=left_primary, baseline=left_baseline,
                      cov_colors=lcc, clip=clip, thick=thick, sort=False,
                      show_ylabels=True, title=left_title,
                      xlabel=left_xlabel if left_xlabel is not None else xlabel,
                      xlim=xlim, autoscale_x=autoscale_x)
    sR = forest_panel(axes[1], recs, primary=right_primary, baseline=right_baseline,
                      cov_colors=rcc, clip=clip, thick=thick, sort=False,
                      show_ylabels=False, title=right_title,
                      xlabel=right_xlabel if right_xlabel is not None else xlabel,
                      xlim=xlim, autoscale_x=autoscale_x)
    if share_x and xlim is None:
        lo = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
        hi = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
        axes[0].set_xlim(lo, hi); axes[1].set_xlim(lo, hi)
    if lcc == rcc:
        legend_handles = _legend_handles(right_primary, right_baseline, rcc, thick)
    else:
        # combined legend: each method's "covered" colour + a shared "missed"
        legend_handles = [
            Line2D([0], [0], color=lcc[0], marker="o", lw=2.4, ms=5,
                   label=f"{left_primary.label} (covered)"),
            Line2D([0], [0], color=rcc[0], marker="o", lw=2.4, ms=5,
                   label=f"{right_primary.label} (covered)"),
            Line2D([0], [0], color=rcc[1], marker="o", lw=2.4, ms=5, label="missed"),
            Line2D([0], [0], color=TRUTH_COLOR, marker=TRUTH_MARKER, ls="none",
                   ms=7, mew=1.5, label=r"true $\theta^*$"),
        ]
    if legend:
        axes[1].legend(handles=legend_handles, loc="lower right", fontsize=11,
                       framealpha=0.95, borderpad=0.8, handlelength=1.6,
                       handletextpad=0.7, borderaxespad=0.6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path
