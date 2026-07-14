"""Generate results.pdf: WCOJ speedup over the binary-join baseline.

Artifact-backed numbers (single-system ablation); see https://xlog.md/guides/benchmarking.
Run: python make_results.py  ->  results.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the paper's Computer Modern typography.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.linewidth": 0.5,
})

INK = "#2d3440"
MUTED = "#5a626e"
BAR = "#1b804f"      # figgreen
ACCENT = "#b02a2a"   # figaccent

# Fixture display names (paper Sec. 8.2), artifact order preserved.
fixtures = [
    ("call-graph edges", 29.62),
    ("Andersen points-to", 26.96),
    ("disassembly (ddisasm)", 28.79),
    ("NeSy mining analog", 26.60),
]
geomean = 27.96  # artifact-backed geometric mean

names = [f[0] for f in fixtures]
speedup = [f[1] for f in fixtures]
ypos = range(len(fixtures) - 1, -1, -1)  # first fixture on top

fig, ax = plt.subplots(figsize=(3.3, 1.55))
ax.barh(ypos, speedup, height=0.58, color=BAR, zorder=2)
ax.axvline(geomean, ls=(0, (4, 3)), lw=0.8, color=ACCENT, zorder=3)

for y, v in zip(ypos, speedup):
    ax.text(v - 0.6, y, f"{v:.2f}$\\times$", ha="right", va="center",
            fontsize=7, color="white", zorder=4,
            bbox=dict(boxstyle="square,pad=0.15", fc=BAR, ec="none"))

ax.text(geomean - 0.5, len(fixtures) - 0.08, f"geomean {geomean:.2f}$\\times$",
        color=ACCENT, fontsize=7, ha="right", va="top", zorder=4)

ax.set_yticks(list(ypos))
ax.set_yticklabels(names, fontsize=7.5, color=INK)
ax.set_xlim(0, 34)
ax.set_ylim(-0.55, len(fixtures) - 0.05)
ax.set_xticks([0, 10, 20, 30])
ax.set_xlabel(r"speedup over binary-join baseline ($\times$)",
              fontsize=7.5, color=INK)
ax.tick_params(axis="both", length=2.5, width=0.5, colors=MUTED,
               labelcolor=INK)
ax.xaxis.grid(True, ls=":", lw=0.4, color="#c8ccd2", zorder=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)

fig.tight_layout(pad=0.3)
fig.savefig("results.pdf")
print("wrote results.pdf")
