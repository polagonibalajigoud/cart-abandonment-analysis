"""Plotting helpers for Step 7 (pure pandas/matplotlib, no Spark needed)."""
import numpy as np
import matplotlib.pyplot as plt

BLUE, RED, GREY = "#2a6f97", "#c1121f", "#8d99ae"


def rate_bar(ax, pdf, title, xlabel, overall, rotate=0):
    """pdf columns: label, sessions, rate (percent)."""
    bars = ax.bar(pdf["label"].astype(str), pdf["rate"], color=BLUE, width=0.65)
    ax.axhline(overall, color=RED, linestyle="--", linewidth=1.3)
    for b, r, n in zip(bars, pdf["rate"], pdf["sessions"]):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}%", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 2, f"n={int(n):,}", ha="center",
                color="white", fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(0, 85)
    ax.tick_params(axis="x", rotation=rotate, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def rate_line(ax, pdf, title, xlabel, overall):
    ax.plot(pdf["label"], pdf["rate"], marker="o", color=BLUE, linewidth=1.8)
    ax.axhline(overall, color=RED, linestyle="--", linewidth=1.3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(0, 85)
    ax.spines[["top", "right"]].set_visible(False)


def confusion_heatmap(ax, cm, title):
    """cm = [[TN, FP], [FN, TP]] with rows = actual (purchased, abandoned)."""
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > (cm.min() + cm.max()) / 2 else "black")
    ax.set_xticks([0, 1], ["Purchased", "Abandoned"])
    ax.set_yticks([0, 1], ["Purchased", "Abandoned"])
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=11)


def importance_barh(ax, names, values, title, top=8):
    pairs = sorted(zip(names, values), key=lambda x: x[1])[-top:]
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=BLUE)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Importance", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def behaviour_dashboard(frames, overall, path):
    """frames: dict with keys device, source, items, value, hour, checkout."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    rate_bar(axes[0, 0], frames["device"], "By device", "Device", overall)
    rate_bar(axes[0, 1], frames["source"], "By traffic source", "Traffic source", overall, rotate=20)
    rate_bar(axes[0, 2], frames["checkout"], "By checkout status", "Checkout status", overall)
    rate_bar(axes[1, 0], frames["items"], "By items added to cart", "Items added", overall)
    rate_bar(axes[1, 1], frames["value"], "By cart value ($)", "Cart value", overall)
    rate_line(axes[1, 2], frames["hour"], "By hour of day", "Hour of day (0-23)", overall)
    fig.suptitle(f"Cart abandonment rate by behaviour   (dashed line = overall {overall:.1f}%)",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    return fig


def model_panel(cm_a, cm_b, imp_a, imp_b, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    confusion_heatmap(axes[0, 0], cm_a, "Model A: all features")
    confusion_heatmap(axes[0, 1], cm_b, "Model B: browsing only")
    importance_barh(axes[1, 0], *imp_a, "Model A: top features")
    importance_barh(axes[1, 1], *imp_b, "Model B: top features")
    fig.suptitle("Decision Tree models: confusion matrices and feature importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    return fig
