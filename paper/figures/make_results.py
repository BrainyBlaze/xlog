"""Generate results.pdf: WCOJ speedup over the binary-join baseline.

Artifact-backed numbers (single-system ablation); see https://xlog.md/guides/benchmarking.
Run: python make_results.py  ->  results.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

fixtures = ["call_graph", "andersen", "ddisasm", "nesy_analog"]
speedup = [29.62, 26.96, 28.79, 26.60]
geomean = 27.96

fig, ax = plt.subplots(figsize=(3.3, 2.1))
bars = ax.bar(range(len(fixtures)), speedup, width=0.62,
              color="#1b804f", edgecolor="black", linewidth=0.5, zorder=2)
ax.axhline(geomean, ls="--", lw=0.8, color="#b02a2a", zorder=1)
for b, v in zip(bars, speedup):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:g}$\\times$",
            ha="center", va="bottom", fontsize=7, zorder=3,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"))
ax.text(0.98, 0.97, f"geomean {geomean:g}$\\times$",
        transform=ax.transAxes, color="#b02a2a",
        fontsize=7, ha="right", va="top", zorder=3,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
ax.set_xticks(range(len(fixtures)))
ax.set_xticklabels(fixtures, rotation=20, ha="right")
ax.set_ylabel(r"WCOJ speedup ($\times$)")
ax.set_ylim(0, 36)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("results.pdf")
print("wrote results.pdf")
