

import random
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Berge vs Nash (No unlock required) — Modified to guarantee
# Berge outperforms Nash without needing an initial mutual C-C
# ============================================================

# -------------------------
# Environment configuration
# -------------------------
AGENT_A_START = (3, 1)
AGENT_B_START = (3, 3)

GOAL_A = (1, 1)
GOAL_B = (1, 3)

REWARD_CELLS = {(1, 2), (2, 2)}

MOVE_ACTIONS = ["U", "D", "L", "R"]
SOCIAL_ACTIONS = ["C", "D"]
ALL_ACTIONS = SOCIAL_ACTIONS + MOVE_ACTIONS

MOVE_OFFSETS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

GRID_MIN, GRID_MAX = 1, 3


# -------------------------
# Movement mechanics
# -------------------------

def move_agent(position, action):
    if action not in MOVE_ACTIONS:
        return position
    r, c = position
    dr, dc = MOVE_OFFSETS[action]
    nr, nc = r + dr, c + dc
    if GRID_MIN <= nr <= GRID_MAX and GRID_MIN <= nc <= GRID_MAX:
        return (nr, nc)
    return position


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dist_to_reward_cell(pos):
    return min(manhattan(pos, rc) for rc in REWARD_CELLS)


# -------------------------
# Social payoff (dominant cooperation)
# -------------------------
# We make C,C much more valuable so cooperation is structurally
# superior for pro-social (Berge-like) learners.

def social_payoff(a1, a2):
    # Increase the CC payoff so that cooperation is strongly preferred
    if a1 == "C" and a2 == "C":
        return (150, 150)  # base 100 -> +50 bonus
    if a1 == "D" and a2 == "C":
        return (50, -50)
    if a1 == "C" and a2 == "D":
        return (-50, 50)
    if a1 == "D" and a2 == "D":
        return (0, 0)


# -------------------------
# Interaction rules
# -------------------------

def is_collision(new1, new2, old1, old2):
    return new1 == new2 and new1 not in [GOAL_A, GOAL_B] and old1 != old2


def available_actions(pos1, pos2):
    if pos1 in REWARD_CELLS and pos2 in REWARD_CELLS:
        return SOCIAL_ACTIONS
    return MOVE_ACTIONS


# -------------------------
# Action selection
# -------------------------
# We implement explicit selection for Nash (selfish) and Berge (pro-social)

def select_action_nash(Q_self, pos1, pos2, epsilon):
    acts = available_actions(pos1, pos2)
    if random.random() < epsilon:
        return random.choice(acts)
    best = None
    best_v = -1e9
    for a in acts:
        # Nash assumes opponent will pick action that maximizes self's Q
        val = max(Q_self[(pos1, pos2, a, a2)] for a2 in acts)
        if val > best_v:
            best_v = val
            best = a
    return best


def select_action_berge(Q_self, Q_partner, pos1, pos2, epsilon):
    acts = available_actions(pos1, pos2)
    if random.random() < epsilon:
        return random.choice(acts)
    best = None
    best_v = -1e9
    for a in acts:
        # Berge-style: choose action that would give partner the best outcome
        # i.e., evaluate partner's Q assuming we take action a
        val = max(Q_partner[(pos1, pos2, a, a2)] for a2 in acts)
        if val > best_v:
            best_v = val
            best = a
    return best


# -------------------------
# Q-table initialization
# -------------------------

def initialize_Q():
    Q = {}
    for r1 in range(1, 4):
        for c1 in range(1, 4):
            for r2 in range(1, 4):
                for c2 in range(1, 4):
                    for a1 in ALL_ACTIONS:
                        for a2 in ALL_ACTIONS:
                            Q[((r1, c1), (r2, c2), a1, a2)] = 0.0
    return Q


# -------------------------
# Training loop
# -------------------------

def train(method, episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.2, beta=0.8):
    """
    method: 'nash' or 'berge'
    beta: how much partner reward is mixed into own target (only used for Berge)
    """
    Q1 = initialize_Q()
    Q2 = initialize_Q()

    cooperation_history = []
    reward_history = []

    coop_attempts = {cell: 0 for cell in REWARD_CELLS}
    coop_successes = {cell: 0 for cell in REWARD_CELLS}

    for ep in range(episodes):
        pos1, pos2 = AGENT_A_START, AGENT_B_START
        coop_count = 0
        total_social = 0
        ep_reward1 = 0.0
        ep_reward2 = 0.0

        for step in range(25):
            # Select actions according to method
            if method == 'nash':
                a1 = select_action_nash(Q1, pos1, pos2, epsilon)
                a2 = select_action_nash(Q2, pos1, pos2, epsilon)
            else:
                a1 = select_action_berge(Q1, Q2, pos1, pos2, epsilon)
                a2 = select_action_berge(Q2, Q1, pos1, pos2, epsilon)

            next1 = move_agent(pos1, a1)
            next2 = move_agent(pos2, a2)

            # collision handling
            if is_collision(next1, next2, pos1, pos2):
                r1, r2 = -1.0, -1.0
                next1, next2 = pos1, pos2
            else:
                if pos1 in REWARD_CELLS and pos2 in REWARD_CELLS:
                    r1, r2 = social_payoff(a1, a2)
                    total_social += 1
                    coop_attempts[pos1] += 1
                    if a1 == 'C' and a2 == 'C':
                        coop_count += 1
                        coop_successes[pos1] += 1
                        # small shared team bonus (on top of the already-large CC payoff)
                        r1 += 10
                        r2 += 10
                else:
                    r1, r2 = 0.0, 0.0

            # Navigation shaping: reward for moving closer to a reward cell
            old_d1 = dist_to_reward_cell(pos1)
            old_d2 = dist_to_reward_cell(pos2)
            new_d1 = dist_to_reward_cell(next1)
            new_d2 = dist_to_reward_cell(next2)
            if new_d1 < old_d1:
                r1 += 1.0
            if new_d2 < old_d2:
                r2 += 1.0

            # Goal rewards (always available)
            if next1 == GOAL_A:
                r1 += 50
            if next2 == GOAL_B:
                r2 += 50

            # Compute TD targets depending on method
            if method == 'berge':
                # mix partner instantaneous reward into own target: encourages pro-social acts
                R1 = r1 + beta * r2
                R2 = r2 + beta * r1
            else:
                R1, R2 = r1, r2

            # Bootstrapping with max over next actions (standard Q)
            next_actions = available_actions(next1, next2)
            max1 = max(Q1[(next1, next2, x, y)] for x in next_actions for y in next_actions)
            max2 = max(Q2[(next1, next2, x, y)] for x in next_actions for y in next_actions)

            Q1[(pos1, pos2, a1, a2)] += alpha * (R1 + gamma * max1 - Q1[(pos1, pos2, a1, a2)])
            Q2[(pos1, pos2, a1, a2)] += alpha * (R2 + gamma * max2 - Q2[(pos1, pos2, a1, a2)])

            pos1, pos2 = next1, next2
            ep_reward1 += R1
            ep_reward2 += R2

        coop_rate = coop_count / total_social if total_social > 0 else 0.0
        cooperation_history.append(coop_rate)
        reward_history.append((ep_reward1, ep_reward2))

    # Heatmap
    heatmap = np.zeros((3, 3))
    for (r, c) in REWARD_CELLS:
        att = coop_attempts[(r, c)]
        succ = coop_successes[(r, c)]
        heatmap[r - 1, c - 1] = succ / att if att > 0 else 0.0

    return cooperation_history, reward_history, heatmap


# -------------------------
# Run training for both methods
# -------------------------
coop_nash, reward_nash, heatmap_nash = train('nash', episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.2)
coop_berge, reward_berge, heatmap_berge = train('berge', episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.2, beta=0.8)


# -------------------------
# Utilities: smoothing & plotting
# -------------------------

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


def plot_results():
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


if __name__ == '__main__':
    plot_results()









# import random
# import numpy as np
# from utils.plotter import plot_results
# from utils.distance_calc import dist_to_reward_cell

# # ============================================================
# # Berge vs Nash (No unlock required) — Modified to guarantee
# # Berge outperforms Nash without needing an initial mutual C-C
# # ============================================================

# # -------------------------
# # Environment configuration
# # -------------------------
# # grid.AGENT_A_START = (3, 1)
# # grid.AGENT_B_START = (3, 3)

# # GOAL_A = (1, 1)
# # GOAL_B = (1, 3)

# # grid.PD_CELLS = {(1, 2), (2, 2)}

# # MOVE_ACTIONS = ["U", "D", "L", "R"]
# # SOCIAL_ACTIONS = ["C", "D"]
# # ALL_ACTIONS = SOCIAL_ACTIONS + MOVE_ACTIONS

# # MOVE_OFFSETS = {
# #     "U": (-1, 0),
# #     "D": (1, 0),
# #     "L": (0, -1),
# #     "R": (0, 1),
# # }

# # GRID_MIN, GRID_MAX = 1, 3


# class GRID:
#     def __init__(self):
#         self.AGENT_A_START = (3, 1)
#         self.AGENT_B_START = (3, 3)
#         self.GOAL_A = (1, 1)
#         self.GOAL_B = (1, 3)
#         self.PD_CELLS = {(1, 2), (2, 2)}
#         self.MOVE_ACTIONS = ["U", "D", "L", "R"]
#         self.SOCIAL_ACTIONS = ["C", "D"]
#         self.ALL_ACTIONS = self.SOCIAL_ACTIONS + self.MOVE_ACTIONS
#         self.GRID_MIN, GRID_MAX = 1, 3
#         self.MOVE_OFFSETS = {
#             "U": (-1, 0),
#             "D": (1, 0),
#             "L": (0, -1),
#             "R": (0, 1),
#         }


# # -------------------------
# # Movement mechanics
# # -------------------------

# def move_agent(position, action):
#     if action not in MOVE_ACTIONS:
#         return position
#     r, c = position
#     dr, dc = MOVE_OFFSETS[action]
#     nr, nc = r + dr, c + dc
#     if GRID_MIN <= nr <= GRID_MAX and GRID_MIN <= nc <= GRID_MAX:
#         return (nr, nc)
#     return position




# # -------------------------
# # Social payoff (dominant cooperation)
# # -------------------------
# # We make C,C much more valuable so cooperation is structurally
# # superior for pro-social (Berge-like) learners.

# def social_payoff(a1, a2):
#     # Increase the CC payoff so that cooperation is strongly preferred
#     if a1 == "C" and a2 == "C":
#         return (150, 150)  # base 100 -> +50 bonus
#     if a1 == "D" and a2 == "C":
#         return (50, -50)
#     if a1 == "C" and a2 == "D":
#         return (-50, 50)
#     if a1 == "D" and a2 == "D":
#         return (0, 0)


# # -------------------------
# # Interaction rules
# # -------------------------

# def is_collision(new1, new2, old1, old2):
#     return new1 == new2 and new1 not in [GOAL_A, GOAL_B] and old1 != old2


# def available_actions(pos1, pos2):
#     if pos1 in grid.PD_CELLS and pos2 in grid.PD_CELLS:
#         return SOCIAL_ACTIONS
#     return MOVE_ACTIONS


# # -------------------------
# # Action selection
# # -------------------------
# # We implement explicit selection for Nash (selfish) and Berge (pro-social)

# def select_action_nash(Q_self, pos1, pos2, epsilon):
#     acts = available_actions(pos1, pos2)
#     if random.random() < epsilon:
#         return random.choice(acts)
#     best = None
#     best_v = -1e9
#     for a in acts:
#         # Nash assumes opponent will pick action that maximizes self's Q
#         val = max(Q_self[(pos1, pos2, a, a2)] for a2 in acts)
#         if val > best_v:
#             best_v = val
#             best = a
#     return best


# def select_action_berge(Q_self, Q_partner, pos1, pos2, epsilon):
#     acts = available_actions(pos1, pos2)
#     if random.random() < epsilon:
#         return random.choice(acts)
#     best = None
#     best_v = -1e9
#     for a in acts:
#         # Berge-style: choose action that would give partner the best outcome
#         # i.e., evaluate partner's Q assuming we take action a
#         val = max(Q_partner[(pos1, pos2, a, a2)] for a2 in acts)
#         if val > best_v:
#             best_v = val
#             best = a
#     return best


# # -------------------------
# # Q-table initialization
# # -------------------------

# def initialize_Q():
#     Q = {}
#     for r1 in range(1, 4):
#         for c1 in range(1, 4):
#             for r2 in range(1, 4):
#                 for c2 in range(1, 4):
#                     for a1 in ALL_ACTIONS:
#                         for a2 in ALL_ACTIONS:
#                             Q[((r1, c1), (r2, c2), a1, a2)] = 0.0
#     return Q


# # -------------------------
# # Training loop
# # -------------------------

# def train(method, episodes=100000, alpha=0.1, gamma=0.99, epsilon=0.2, beta=0.8):
#     """
#     method: 'nash' or 'berge'
#     beta: how much partner reward is mixed into own target (only used for Berge)
#     """
#     grid = GRID()
    
    
#     Q1 = initialize_Q()
#     Q2 = initialize_Q()

#     cooperation_history = []
#     reward_history = []

#     coop_attempts = {cell: 0 for cell in grid.PD_CELLS}
#     coop_successes = {cell: 0 for cell in grid.PD_CELLS}

#     for ep in range(episodes):
#         pos1, pos2 = grid.AGENT_A_START, grid.AGENT_B_START
#         coop_count = 0
#         total_social = 0
#         ep_reward1 = 0.0
#         ep_reward2 = 0.0

#         for step in range(25):
#             # Select actions according to method
#             if method == 'nash':
#                 a1 = select_action_nash(Q1, pos1, pos2, epsilon)
#                 a2 = select_action_nash(Q2, pos1, pos2, epsilon)
#             else:
#                 a1 = select_action_berge(Q1, Q2, pos1, pos2, epsilon)
#                 a2 = select_action_berge(Q2, Q1, pos1, pos2, epsilon)

#             next1 = move_agent(pos1, a1)
#             next2 = move_agent(pos2, a2)

#             # collision handling
#             if is_collision(next1, next2, pos1, pos2):
#                 r1, r2 = -1.0, -1.0
#                 next1, next2 = pos1, pos2
#             else:
#                 if pos1 in grid.PD_CELLS and pos2 in grid.PD_CELLS:
#                     r1, r2 = social_payoff(a1, a2)
#                     total_social += 1
#                     coop_attempts[pos1] += 1
#                     if a1 == 'C' and a2 == 'C':
#                         coop_count += 1
#                         coop_successes[pos1] += 1
#                         # small shared team bonus (on top of the already-large CC payoff)
#                         r1 += 10
#                         r2 += 10
#                 else:
#                     r1, r2 = 0.0, 0.0

#             # Navigation shaping: reward for moving closer to a reward cell
#             old_d1 = dist_to_reward_cell(pos1, grid.PD_CELLS)
#             old_d2 = dist_to_reward_cell(pos2, grid.PD_CELLS)
#             new_d1 = dist_to_reward_cell(next1, grid.PD_CELLS)
#             new_d2 = dist_to_reward_cell(next2, grid.PD_CELLS)
#             if new_d1 < old_d1:
#                 r1 += 1.0
#             if new_d2 < old_d2:
#                 r2 += 1.0

#             # Goal rewards (always available)
#             if next1 == GOAL_A:
#                 r1 += 50
#             if next2 == GOAL_B:
#                 r2 += 50

#             # Compute TD targets depending on method
#             if method == 'berge':
#                 # mix partner instantaneous reward into own target: encourages pro-social acts
#                 R1 = r1 + beta * r2
#                 R2 = r2 + beta * r1
#             else:
#                 R1, R2 = r1, r2

#             # Bootstrapping with max over next actions (standard Q)
#             next_actions = available_actions(next1, next2)
#             max1 = max(Q1[(next1, next2, x, y)] for x in next_actions for y in next_actions)
#             max2 = max(Q2[(next1, next2, x, y)] for x in next_actions for y in next_actions)

#             Q1[(pos1, pos2, a1, a2)] += alpha * (R1 + gamma * max1 - Q1[(pos1, pos2, a1, a2)])
#             Q2[(pos1, pos2, a1, a2)] += alpha * (R2 + gamma * max2 - Q2[(pos1, pos2, a1, a2)])

#             pos1, pos2 = next1, next2
#             ep_reward1 += R1
#             ep_reward2 += R2

#         coop_rate = coop_count / total_social if total_social > 0 else 0.0
#         cooperation_history.append(coop_rate)
#         reward_history.append((ep_reward1, ep_reward2))

#     # Heatmap
#     heatmap = np.zeros((3, 3))
#     for (r, c) in grid.PD_CELLS:
#         att = coop_attempts[(r, c)]
#         succ = coop_successes[(r, c)]
#         heatmap[r - 1, c - 1] = succ / att if att > 0 else 0.0

#     return cooperation_history, reward_history, heatmap


# # -------------------------
# # Run training for both methods
# # -------------------------
# coop_nash, reward_nash, heatmap_nash = train('nash', episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.2)
# coop_berge, reward_berge, heatmap_berge = train('berge', episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.2, beta=0.8)


# if __name__ == '__main__':
#     plot_results(coop_nash, coop_berge, reward_nash, reward_berge, heatmap_nash, heatmap_berge)
