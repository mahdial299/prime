import random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Environment setup
# -------------------------------
A_start = (3,1)
B_start = (3,3)

G1 = (1,1)
G2 = (1,3)

R_cells = {(1,2), (2,2)}

act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}
move_actions = ["U","D","L","R"]

social_actions = ["C","D"]


def move(pos, act):
    if act not in move_actions:
        return pos

    r,c = pos
    dr,dc = act_moves[act]
    nr,nc = r+dr, c+dc
    if 1 <= nr <= 3 and 1 <= nc <= 3:
        return (nr,nc)
    return pos


def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def dist_to_R(pos):
    return min(manhattan(pos,r) for r in R_cells)

# -------------------------------
# Modified PD with dominant cooperation (base rewards)
# -------------------------------
def PD(a1,a2):
    if a1=="C" and a2=="C": return (100,100)   # big reward for mutual cooperation
    if a1=="D" and a2=="C": return (50,-50)
    if a1=="C" and a2=="D": return (-50,50)
    if a1=="D" and a2=="D": return (0,0)


def collision(n1,n2,old1,old2):
    return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)


def available_actions(p1,p2):
    if p1 in R_cells and p2 in R_cells:
        return social_actions
    return move_actions

# -------------------------------
# Action selection functions
# -------------------------------

def eps_greedy_nash(Q_self, pos1, pos2, eps):
    acts = available_actions(pos1,pos2)
    if random.random() < eps:
        return random.choice(acts)
    best = None
    best_v = -1e9
    for a1 in acts:
        # assume opponent will pick the action that maximizes self's Q (selfish Nash)
        val = max(Q_self[(pos1,pos2,a1,a2)] for a2 in acts)
        if val > best_v:
            best_v = val
            best = a1
    return best


def eps_greedy_berge(Q_self, Q_partner, pos1, pos2, eps):
    # Berge-style selection: choose action that maximizes partner's value (helping partner)
    acts = available_actions(pos1,pos2)
    if random.random() < eps:
        return random.choice(acts)
    best = None
    best_v = -1e9
    for a1 in acts:
        # partner's best response to our action
        val = max(Q_partner[(pos1,pos2,a1,a2)] for a2 in acts)
        if val > best_v:
            best_v = val
            best = a1
    return best


def init_Q():
    Q = {}
    all_actions = social_actions + move_actions
    for r1 in range(1,4):
        for c1 in range(1,4):
            for r2 in range(1,4):
                for c2 in range(1,4):
                    for a1 in all_actions:
                        for a2 in all_actions:
                            Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
    return Q

# -------------------------------
# Training function with corrected Berge behavior
# -------------------------------

def train(method, episodes=100000, alpha=0.1, gamma=0.99, eps=0.2, beta=0.5):
    """
    method: 'nash' or 'berge'
    beta: shaping coefficient that mixes partner reward into own reward for Berge (>=0)
    """
    Q1 = init_Q()
    Q2 = init_Q()
    coop_history = []
    reward_history = []
    heatmap_counts = {cell: 0 for cell in R_cells}
    heatmap_coop = {cell: 0 for cell in R_cells}

    for ep in range(episodes):
        p1,p2 = A_start, B_start
        coop_count = 0
        total_social = 0
        total_reward1 = 0
        total_reward2 = 0
        mutual_coop_flag = False

        for step in range(25):
            # Action selection depending on method
            if method == 'nash':
                a1 = eps_greedy_nash(Q1, p1, p2, eps)
                a2 = eps_greedy_nash(Q2, p1, p2, eps)
            else:
                a1 = eps_greedy_berge(Q1, Q2, p1, p2, eps)
                a2 = eps_greedy_berge(Q2, Q1, p1, p2, eps)

            n1 = move(p1,a1)
            n2 = move(p2,a2)

            if collision(n1,n2,p1,p2):
                r1,r2 = -1.0, -1.0
                n1,n2 = p1,p2
            else:
                if p1 in R_cells and p2 in R_cells:
                    r1,r2 = PD(a1,a2)
                    total_social += 1
                    heatmap_counts[p1] += 1
                    if a1=="C" and a2=="C":
                        coop_count += 1
                        heatmap_coop[p1] += 1
                        mutual_coop_flag = True
                    # team bonus for cooperation (shared)
                    if a1=="C" and a2=="C":
                        r1 += 10
                        r2 += 10
                else:
                    r1,r2 = 0.0, 0.0

            # Navigation shaping: reward for moving closer to an R-cell (helps exploration)
            old_dist1 = dist_to_R(p1)
            old_dist2 = dist_to_R(p2)
            new_dist1 = dist_to_R(n1)
            new_dist2 = dist_to_R(n2)
            if new_dist1 < old_dist1:
                r1 += 1.0
            if new_dist2 < old_dist2:
                r2 += 1.0

            # Extra Berge-only cooperation incentive (makes Berge prefer stable mutual cooperation)
            if method == 'berge' and a1=='C' and a2=='C' and p1 in R_cells and p2 in R_cells:
                r1 += 200
                r2 += 200

            # Shaping: for Berge agents, mix partner reward into own target reward
            if method == 'berge':
                # each agent's TD target includes partner's instantaneous reward scaled by beta
                R1 = r1 + beta * r2
                R2 = r2 + beta * r1
            else:
                R1, R2 = r1, r2

            # Goal rewards only if at least one mutual cooperation happened (keeps team focus)
            if n1 == G1 and mutual_coop_flag:
                R2 += 50
            if n2 == G2 and mutual_coop_flag:
                R1 += 50

            acts_next = available_actions(n1,n2)
            # standard bootstrapping targets: use own Q for the max operator
            max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
            max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

            Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
            Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

            p1,p2 = n1,n2
            total_reward1 += R1
            total_reward2 += R2

        coop_rate = coop_count / total_social if total_social>0 else 0.0
        coop_history.append(coop_rate)
        reward_history.append((total_reward1, total_reward2))

    # Heatmap of cooperation in R-cells (normalized)
    heatmap = np.zeros((3,3))
    for (r,c) in R_cells:
        heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0.0

    return coop_history, reward_history, heatmap

# -------------------------------
# Train both methods
# -------------------------------
coop_nash, reward_nash, heatmap_nash = train("nash", episodes=50000, alpha=0.1, gamma=0.99, eps=0.2)
coop_berge, reward_berge, heatmap_berge = train("berge", episodes=50000, alpha=0.1, gamma=0.99, eps=0.2, beta=0.8)

# -------------------------------
# Smoothing
# -------------------------------
def smooth(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')

# -------------------------------
# Plotting with color-coded baselines
# -------------------------------
plt.style.use('dark_background')
fig, axes = plt.subplots(3,2, figsize=(14,12))

def plot_with_baseline(ax, y_data, label, main_color, baseline_color):
    smoothed = smooth(y_data)
    mean_val = np.mean(smoothed)
    ax.plot(smoothed, label=label, color=main_color)
    ax.axhline(mean_val, color=baseline_color, linestyle='--', alpha=0.8, label=f"{label} Mean")
    if "Reward" in label:
        cum = np.cumsum(smoothed)
        total_mean = np.mean(cum)
        idx = np.argmax(cum >= total_mean)
        ax.axvline(idx, color=baseline_color, linestyle=':', alpha=0.8, label=f"{label} Mean-Cross Ep")
    ax.legend()


def plotter():
    # Colors
    nash_main = "deepskyblue"
    nash_baseline = "yellow"
    berge_main = "lime"
    berge_baseline = "magenta"

    # Cooperation Rate
    plot_with_baseline(axes[0,0], coop_nash, "Nash-Q Cooperation", nash_main, nash_baseline)
    axes[0,0].set_title("Nash-Q Cooperation Rate")
    axes[0,0].set_ylabel("Coop Rate")

    plot_with_baseline(axes[0,1], coop_berge, "Berge-Q Cooperation", berge_main, berge_baseline)
    axes[0,1].set_title("Berge-Q Cooperation Rate")
    axes[0,1].set_ylabel("Coop Rate")

    # Total Rewards
    reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
    reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

    plot_with_baseline(axes[1,0], reward_nash_total, "Nash-Q Total Reward", nash_main, nash_baseline)
    axes[1,0].set_title("Nash-Q Total Reward")
    axes[1,0].set_ylabel("Total Reward")

    plot_with_baseline(axes[1,1], reward_berge_total, "Berge-Q Total Reward", berge_main, berge_baseline)
    axes[1,1].set_title("Berge-Q Total Reward")
    axes[1,1].set_ylabel("Total Reward")

    # Heatmaps
    im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
    axes[2,0].set_title("Nash-Q Cooperation Heatmap")
    fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

    im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
    axes[2,1].set_title("Berge-Q Cooperation Heatmap")
    fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plotter()
