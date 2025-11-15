# -------------------------
# Utilities: smoothing & plotting
# -------------------------
import numpy as np
import matplotlib.pyplot as plt


def smooth(values, window=100):
    return np.convolve(values, np.ones(window) / window, mode='valid')

plt.style.use('dark_background')
fig, axes = plt.subplots(3, 2, figsize=(14, 12))


def plot_with_baseline(ax, data, label, color_main, color_baseline):
    sm = smooth(data)
    mean_val = np.mean(sm)
    ax.plot(sm, color=color_main, label=label)
    ax.axhline(mean_val, color=color_baseline, linestyle='--', alpha=0.8, label=f"{label} Mean")
    if 'Reward' in label:
        cum = np.cumsum(sm)
        idx = np.argmax(cum >= np.mean(cum))
        ax.axvline(idx, color=color_baseline, linestyle=':', alpha=0.8, label=f"{label} Mean-Cross Ep")
    ax.legend()




def plot_results(coop_nash, coop_berge, reward_nash, reward_berge, heatmap_nash, heatmap_berge):
    COLORS = {
        'nash_main': 'deepskyblue',
        'nash_base': 'yellow',
        'berge_main': 'lime',
        'berge_base': 'magenta'
    }

    plot_with_baseline(axes[0, 0], coop_nash, 'Nash-Q Cooperation', COLORS['nash_main'], COLORS['nash_base'])
    axes[0, 0].set_title('Nash-Q Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')

    plot_with_baseline(axes[0, 1], coop_berge, 'Berge-Q Cooperation', COLORS['berge_main'], COLORS['berge_base'])
    axes[0, 1].set_title('Berge-Q Cooperation Rate')
    axes[0, 1].set_ylabel('Cooperation Rate')

    total_nash = [r1 + r2 for r1, r2 in reward_nash]
    total_berge = [r1 + r2 for r1, r2 in reward_berge]

    plot_with_baseline(axes[1, 0], total_nash, 'Nash-Q Total Reward', COLORS['nash_main'], COLORS['nash_base'])
    axes[1, 0].set_title('Nash-Q Total Reward')
    axes[1, 0].set_ylabel('Total Reward')

    plot_with_baseline(axes[1, 1], total_berge, 'Berge-Q Total Reward', COLORS['berge_main'], COLORS['berge_base'])
    axes[1, 1].set_title('Berge-Q Total Reward')
    axes[1, 1].set_ylabel('Total Reward')

    im0 = axes[2, 0].imshow(heatmap_nash, cmap='Blues', vmin=0, vmax=1)
    axes[2, 0].set_title('Nash-Q Cooperation Heatmap')
    fig.colorbar(im0, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im1 = axes[2, 1].imshow(heatmap_berge, cmap='Greens', vmin=0, vmax=1)
    axes[2, 1].set_title('Berge-Q Cooperation Heatmap')
    fig.colorbar(im1, ax=axes[2, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()