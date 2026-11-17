from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from src.paths import RESULT_DIR

scan_dir = RESULT_DIR / "scan"
out_dir = RESULT_DIR

d_list = [0.3, 0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.98, 0.99, 0.999]


@dataclass
class Case:
    slug: str
    ours: list[float]
    truth: float
    ref: float | None = None
    params: list[float] = field(default_factory=lambda: list(d_list))


c0 = Case(
    slug="defaultIV",
    ours=[180.0] + [175.0] * 3 + [170.0] * 8,
    truth=170.0,
)
c1 = Case(
    slug="defaultIII",
    ours=[80.0] * 2 + [75.0] * 5 + [70.0] * 4 + [60.0],
    truth=73.0,
)
c2 = Case(slug="kangkai_2022_case1", ours=[70.0] * 12, truth=74.5, ref=73.0)
c3 = Case(slug="kangkai_2022_case2", ours=[70.0] * 12, truth=76.0, ref=73.5)
c4 = Case(slug="kangkai_2022_case3", ours=[75.0] * 12, truth=83.0, ref=80.0)
c5 = Case(slug="liruiqi_2025_caseA", ours=[70.0] * 12, truth=62.85, ref=59.60)
c6 = Case(
    slug="liruiqi_2025_caseB", ours=[85.0] * 2 + [80.0] * 10, truth=49.21, ref=56.80
)
c7 = Case(slug="liruiqi_2025_caseC", ours=[60.0] + [55.0] * 11, truth=52.95, ref=56.40)
c8 = Case(slug="louxueying_2025", ours=[0.0] * 12, truth=180.2, ref=164.0)


c_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

STYLE = {
    "ours": {"color": "#1f77b4", "lw": 1.8},
    "truth": {"color": "#d62728", "lw": 1.4, "ls": "-"},
    "ref": {"color": "#7f7f7f", "lw": 1.2, "ls": "--"},
    "band": {"color": "#1f77b4", "alpha": 0.15, "lw": 0},
}

# 默认 figsize=(12.5, 9.5)
def plot_case_matrix(c_list, ncols=3, figsize=(20, 9), max_xticks=6):
    nrows = -(-len(c_list) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax, case in zip(axes.flat, c_list):
        d = np.asarray(case.params, float)
        ours = np.asarray(case.ours, float)
        x = np.arange(d.size)

        ax.step(x, ours, where="post", color="#1f77b4", lw=1.8, label="ours")

        if case.truth is not None and np.isfinite(case.truth):
            ax.axhline(case.truth, color="#d62728", lw=1.4, label="truth")
            ax.fill_between(
                x, ours, case.truth, step="post", color="#1f77b4", alpha=0.15, lw=0
            )
            ae = ours - case.truth
            re = np.abs(ae / case.truth * 100)
            ax.text(
                0.98,
                0.97,
                f"AE∈[{ae.min():+.1f},{ae.max():+.1f}]\n|RE|max={re.max():.1f}%",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=12,
                bbox={"fc": "white", "ec": "0.8", "lw": 0.5},
            )

        if case.ref is not None and np.isfinite(case.ref):
            ax.axhline(case.ref, color="#7f7f7f", lw=1.2, ls="--", label="ref (paper)")

        ax.set_title(case.slug, fontsize=14)
        idx = np.unique(np.linspace(0, d.size - 1, min(d.size, max_xticks)).astype(int))
        ax.set_xticks(idx)
        ax.set_xticklabels(
            [f"{d[i]:g}" for i in idx], rotation=45, ha="right", fontsize=8
        )
        ax.set_xlim(-0.4, d.size - 0.6)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax.margins(y=0.15)

    for ax in axes.flat[len(c_list) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("pressure")
    for ax in axes[-1, :]:
        ax.set_xlabel("d (evenly spaced)")

    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
        fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


# 用法
if __name__ == "__main__":
    fig, axes = plot_case_matrix(c_list)
    fig.savefig(out_dir / "bp_matrix.png", dpi=200, bbox_inches="tight")
