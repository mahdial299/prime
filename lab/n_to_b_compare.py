# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # Environment setup
# A_start = (3,1)
# B_start = (3,3)
# G1 = (1,1)
# G2 = (1,3)
# R_cells = {(1,2), (2,2)}
# move_actions = ["U","D","L","R"]
# social_actions = ["C","D"]
# act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

# def move(pos, act):
#     if act not in move_actions:
#         return pos
#     r,c = pos
#     dr,dc = act_moves[act]
#     nr,nc = r+dr, c+dc
#     if 1 <= nr <= 3 and 1 <= nc <= 3:
#         return (nr,nc)
#     return pos

# def PD(a1,a2):
#     if a1=="C" and a2=="C": return (3,3)
#     if a1=="D" and a2=="C": return (5,0)
#     if a1=="C" and a2=="D": return (0,5)
#     if a1=="D" and a2=="D": return (1,1)

# def collision(n1,n2,old1,old2):
#     return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

# def available_actions(p1,p2):
#     if p1 in R_cells and p2 in R_cells:
#         return social_actions
#     return move_actions

# def eps_greedy(Q, pos1, pos2, eps):
#     acts = available_actions(pos1,pos2)
#     if random.random()<eps:
#         return random.choice(acts)
#     best=None
#     best_v=-1e9
#     for a1 in acts:
#         val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
#         if val>best_v:
#             best_v=val; best=a1
#     return best

# def init_Q():
#     Q = {}
#     for r1 in range(1,4):
#         for c1 in range(1,4):
#             for r2 in range(1,4):
#                 for c2 in range(1,4):
#                     for a1 in social_actions + move_actions:
#                         for a2 in social_actions + move_actions:
#                             Q[((r1,c1),(r2,c2),a1,a2)] = 0.0  # correct 4-tuple key
#     return Q


# def train(method, episodes=20000, alpha=0.1, gamma=0.99, eps=0.2):
#     Q1 = init_Q()
#     Q2 = init_Q()
#     coop_history = []
#     reward_history = []
#     heatmap_counts = {cell: 0 for cell in R_cells}
#     heatmap_coop = {cell: 0 for cell in R_cells}

#     for ep in range(episodes):
#         p1,p2 = A_start, B_start
#         coop_count = 0
#         total_social = 0
#         total_reward1 = 0
#         total_reward2 = 0

#         for step in range(25):
#             a1 = eps_greedy(Q1,p1,p2,eps)
#             a2 = eps_greedy(Q2,p1,p2,eps)
#             n1 = move(p1,a1)
#             n2 = move(p2,a2)

#             if collision(n1,n2,p1,p2):
#                 r1,r2=-1,-1
#                 n1,n2=p1,p2
#             else:
#                 if p1 in R_cells and p2 in R_cells:
#                     r1,r2 = PD(a1,a2)
#                     total_social += 1
#                     heatmap_counts[p1] += 1
#                     if a1=="C" and a2=="C":
#                         coop_count += 1
#                         heatmap_coop[p1] += 1
#                 else:
#                     r1,r2 = 0,0

#             if method=="berge":
#                 R1,R2 = r2,r1
#             else:
#                 R1,R2 = r1,r2

#             if n1==G1: R2+=100; break
#             if n2==G2: R1+=100; break

#             acts_next = available_actions(n1,n2)
#             max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
#             max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

#             Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
#             Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

#             p1,p2=n1,n2
#             total_reward1 += R1
#             total_reward2 += R2

#         coop_rate = coop_count / total_social if total_social>0 else 0
#         coop_history.append(coop_rate)
#         reward_history.append((total_reward1, total_reward2))

#     # Compute heatmap rates
#     heatmap = np.zeros((3,3))
#     for (r,c) in R_cells:
#         heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

#     return coop_history, reward_history, heatmap

# # Train
# coop_nash, reward_nash, heatmap_nash = train("nash")
# coop_berge, reward_berge, heatmap_berge = train("berge")

# # Smoothing function
# def smooth(x, w=100):
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# # Plot
# plt.style.use('dark_background')

# fig, axes = plt.subplots(3,2, figsize=(14,12))

# # Cooperation Rate
# axes[0,0].plot(smooth(coop_nash), label="Nash", color="blue")
# axes[0,0].set_title("Nash-Q Smoothed Cooperation Rate")
# axes[0,0].set_xlabel("Episode")
# axes[0,0].set_ylabel("Cooperation Rate")
# axes[0,0].legend()

# axes[0,1].plot(smooth(coop_berge), label="Berge", color="green")
# axes[0,1].set_title("Berge-Q Smoothed Cooperation Rate")
# axes[0,1].set_xlabel("Episode")
# axes[0,1].set_ylabel("Cooperation Rate")
# axes[0,1].legend()

# # Rewards
# reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
# reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

# axes[1,0].plot(smooth(reward_nash_total), label="Nash Total Reward", color="blue")
# axes[1,0].set_title("Nash-Q Total Reward")
# axes[1,0].set_xlabel("Episode")
# axes[1,0].set_ylabel("Total Reward")
# axes[1,0].legend()

# axes[1,1].plot(smooth(reward_berge_total), label="Berge Total Reward", color="green")
# axes[1,1].set_title("Berge-Q Total Reward")
# axes[1,1].set_xlabel("Episode")
# axes[1,1].set_ylabel("Total Reward")
# axes[1,1].legend()

# # Heatmaps
# im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
# axes[2,0].set_title("Nash-Q Cooperation Heatmap in R-cells")
# fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

# im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
# axes[2,1].set_title("Berge-Q Cooperation Heatmap in R-cells")
# fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()



























































# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # Environment setup
# # -------------------------------
# A_start = (3,1)
# B_start = (3,3)
# G1 = (1,1)
# G2 = (1,3)
# R_cells = {(1,2), (2,2)}
# move_actions = ["U","D","L","R"]
# social_actions = ["C","D"]
# act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

# def move(pos, act):
#     if act not in move_actions:
#         return pos
#     r,c = pos
#     dr,dc = act_moves[act]
#     nr,nc = r+dr, c+dc
#     if 1 <= nr <= 3 and 1 <= nc <= 3:
#         return (nr,nc)
#     return pos

# def PD(a1,a2):
#     if a1=="C" and a2=="C": return (3,3)
#     if a1=="D" and a2=="C": return (5,0)
#     if a1=="C" and a2=="D": return (0,5)
#     if a1=="D" and a2=="D": return (1,1)

# def collision(n1,n2,old1,old2):
#     return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

# def available_actions(p1,p2):
#     if p1 in R_cells and p2 in R_cells:
#         return social_actions
#     return move_actions

# def eps_greedy(Q, pos1, pos2, eps):
#     acts = available_actions(pos1,pos2)
#     if random.random()<eps:
#         return random.choice(acts)
#     best=None
#     best_v=-1e9
#     for a1 in acts:
#         val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
#         if val>best_v:
#             best_v=val
#             best=a1
#     return best

# def init_Q():
#     Q = {}
#     for r1 in range(1,4):
#         for c1 in range(1,4):
#             for r2 in range(1,4):
#                 for c2 in range(1,4):
#                     for a1 in social_actions + move_actions:
#                         for a2 in social_actions + move_actions:
#                             Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
#     return Q

# # -------------------------------
# # Training function
# # -------------------------------
# # 20000
# def train(method, episodes=50000, alpha=0.1, gamma=0.99, eps=0.2):
#     Q1 = init_Q()
#     Q2 = init_Q()
#     coop_history = []
#     reward_history = []
#     heatmap_counts = {cell: 0 for cell in R_cells}
#     heatmap_coop = {cell: 0 for cell in R_cells}

#     for ep in range(episodes):
#         p1,p2 = A_start, B_start
#         coop_count = 0
#         total_social = 0
#         total_reward1 = 0
#         total_reward2 = 0

#         for step in range(25):
#             a1 = eps_greedy(Q1,p1,p2,eps)
#             a2 = eps_greedy(Q2,p1,p2,eps)
#             n1 = move(p1,a1)
#             n2 = move(p2,a2)

#             if collision(n1,n2,p1,p2):
#                 r1,r2=-1,-1
#                 n1,n2=p1,p2
#             else:
#                 if p1 in R_cells and p2 in R_cells:
#                     r1,r2 = PD(a1,a2)
#                     total_social += 1
#                     heatmap_counts[p1] += 1
#                     if a1=="C" and a2=="C":
#                         coop_count += 1
#                         heatmap_coop[p1] += 1
#                 else:
#                     r1,r2 = 0,0

#             if method=="berge":
#                 R1,R2 = r2,r1
#             else:
#                 R1,R2 = r1,r2

#             if n1==G1: R2+=100; break
#             if n2==G2: R1+=100; break

#             acts_next = available_actions(n1,n2)
#             max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
#             max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

#             Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
#             Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

#             p1,p2=n1,n2
#             total_reward1 += R1
#             total_reward2 += R2

#         coop_rate = coop_count / total_social if total_social>0 else 0
#         coop_history.append(coop_rate)
#         reward_history.append((total_reward1, total_reward2))

#     # Heatmap of cooperation in R-cells
#     heatmap = np.zeros((3,3))
#     for (r,c) in R_cells:
#         heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

#     return coop_history, reward_history, heatmap

# # -------------------------------
# # Train both methods
# # -------------------------------
# coop_nash, reward_nash, heatmap_nash = train("nash")
# coop_berge, reward_berge, heatmap_berge = train("berge")

# # -------------------------------
# # Smoothing function
# # -------------------------------
# def smooth(x, w=100):
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# # -------------------------------
# # Plotting combined visualization
# # -------------------------------
# plt.style.use('dark_background')
# fig, axes = plt.subplots(3,2, figsize=(14,12))

# # 1️⃣ Cooperation Rate
# axes[0,0].plot(smooth(coop_nash), label="Nash-Q Cooperation Rate", color="blue")
# axes[0,0].set_title("Nash-Q Cooperation Rate")
# axes[0,0].set_ylabel("Coop Rate")
# axes[0,0].legend()

# axes[0,1].plot(smooth(coop_berge), label="Berge-Q Cooperation Rate", color="green")
# axes[0,1].set_title("Berge-Q Cooperation Rate")
# axes[0,1].set_ylabel("Coop Rate")
# axes[0,1].legend()

# # 2️⃣ Total Rewards
# reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
# reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

# axes[1,0].plot(smooth(reward_nash_total), label="Nash-Q Total Reward", color="cyan")
# axes[1,0].set_title("Nash-Q Total Reward")
# axes[1,0].set_ylabel("Total Reward")
# axes[1,0].legend()

# axes[1,1].plot(smooth(reward_berge_total), label="Berge-Q Total Reward", color="lime")
# axes[1,1].set_title("Berge-Q Total Reward")
# axes[1,1].set_ylabel("Total Reward")
# axes[1,1].legend()

# # 3️⃣ Heatmaps of cooperation in R-cells
# im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
# axes[2,0].set_title("Nash-Q Cooperation Heatmap")
# fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

# im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
# axes[2,1].set_title("Berge-Q Cooperation Heatmap")
# fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()

# # -------------------------------
# # Optional: Combined cooperation + cumulative reward
# # -------------------------------
# cumulative_nash = np.cumsum(reward_nash_total)
# cumulative_berge = np.cumsum(reward_berge_total)

# plt.figure(figsize=(12,6))
# plt.plot(smooth(coop_nash), label="Nash-Q Cooperation Rate", color="blue")
# plt.plot(smooth(coop_berge), label="Berge-Q Cooperation Rate", color="green")
# plt.plot(smooth(cumulative_nash/100), label="Nash-Q Cumulative Reward (scaled)", color="cyan")
# plt.plot(smooth(cumulative_berge/100), label="Berge-Q Cumulative Reward (scaled)", color="lime")
# plt.xlabel("Episode")
# plt.ylabel("Cooperation Rate / Cumulative Reward (scaled)")
# plt.title("Cooperation Rate and Cumulative Rewards: Nash-Q vs Berge-Q")
# plt.legend()
# plt.tight_layout()
# plt.show()




















# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # Environment setup
# # -------------------------------
# A_start = (3,1)
# B_start = (3,3)
# G1 = (1,1)
# G2 = (1,3)
# R_cells = {(1,2), (2,2)}
# move_actions = ["U","D","L","R"]
# social_actions = ["C","D"]
# act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

# def move(pos, act):
#     if act not in move_actions:
#         return pos
#     r,c = pos
#     dr,dc = act_moves[act]
#     nr,nc = r+dr, c+dc
#     if 1 <= nr <= 3 and 1 <= nc <= 3:
#         return (nr,nc)
#     return pos

# def PD(a1,a2):
#     if a1=="C" and a2=="C": return (3,3)
#     if a1=="D" and a2=="C": return (5,0)
#     if a1=="C" and a2=="D": return (0,5)
#     if a1=="D" and a2=="D": return (1,1)

# def collision(n1,n2,old1,old2):
#     return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

# def available_actions(p1,p2):
#     if p1 in R_cells and p2 in R_cells:
#         return social_actions
#     return move_actions

# def eps_greedy(Q, pos1, pos2, eps):
#     acts = available_actions(pos1,pos2)
#     if random.random()<eps:
#         return random.choice(acts)
#     best=None
#     best_v=-1e9
#     for a1 in acts:
#         val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
#         if val>best_v:
#             best_v=val
#             best=a1
#     return best

# def init_Q():
#     Q = {}
#     for r1 in range(1,4):
#         for c1 in range(1,4):
#             for r2 in range(1,4):
#                 for c2 in range(1,4):
#                     for a1 in social_actions + move_actions:
#                         for a2 in social_actions + move_actions:
#                             Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
#     return Q

# # -------------------------------
# # Training function
# # -------------------------------
# def train(method, episodes=50000, alpha=0.1, gamma=0.99, eps=0.2):
#     Q1 = init_Q()
#     Q2 = init_Q()
#     coop_history = []
#     reward_history = []
#     heatmap_counts = {cell: 0 for cell in R_cells}
#     heatmap_coop = {cell: 0 for cell in R_cells}

#     for ep in range(episodes):
#         p1,p2 = A_start, B_start
#         coop_count = 0
#         total_social = 0
#         total_reward1 = 0
#         total_reward2 = 0

#         for step in range(25):
#             a1 = eps_greedy(Q1,p1,p2,eps)
#             a2 = eps_greedy(Q2,p1,p2,eps)
#             n1 = move(p1,a1)
#             n2 = move(p2,a2)

#             if collision(n1,n2,p1,p2):
#                 r1,r2=-1,-1
#                 n1,n2=p1,p2
#             else:
#                 if p1 in R_cells and p2 in R_cells:
#                     r1,r2 = PD(a1,a2)
#                     total_social += 1
#                     heatmap_counts[p1] += 1
#                     if a1=="C" and a2=="C":
#                         coop_count += 1
#                         heatmap_coop[p1] += 1
#                 else:
#                     r1,r2 = 0,0

#             if method=="berge":
#                 R1,R2 = r2,r1
#             else:
#                 R1,R2 = r1,r2

#             if n1==G1: R2+=100; break
#             if n2==G2: R1+=100; break

#             acts_next = available_actions(n1,n2)
#             max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
#             max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

#             Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
#             Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

#             p1,p2=n1,n2
#             total_reward1 += R1
#             total_reward2 += R2

#         coop_rate = coop_count / total_social if total_social>0 else 0
#         coop_history.append(coop_rate)
#         reward_history.append((total_reward1, total_reward2))

#     # Heatmap of cooperation in R-cells
#     heatmap = np.zeros((3,3))
#     for (r,c) in R_cells:
#         heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

#     return coop_history, reward_history, heatmap

# # -------------------------------
# # Train both methods
# # -------------------------------
# coop_nash, reward_nash, heatmap_nash = train("nash")
# coop_berge, reward_berge, heatmap_berge = train("berge")

# # -------------------------------
# # Smoothing function
# # -------------------------------
# def smooth(x, w=100):
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# # -------------------------------
# # Plotting with baseline (mean) lines
# # -------------------------------
# plt.style.use('dark_background')
# fig, axes = plt.subplots(3,2, figsize=(14,12))

# def plot_with_baseline(ax, y_data, label, color):
#     smoothed = smooth(y_data)
#     mean_val = np.mean(smoothed)
#     ax.plot(smoothed, label=label, color=color)
#     ax.axhline(mean_val, color=color, linestyle='--', alpha=0.7, label=f"{label} Mean")
#     ax.legend()

# # 1️⃣ Cooperation Rate
# plot_with_baseline(axes[0,0], coop_nash, "Nash-Q Cooperation", "blue")
# axes[0,0].set_title("Nash-Q Cooperation Rate")
# axes[0,0].set_ylabel("Coop Rate")

# plot_with_baseline(axes[0,1], coop_berge, "Berge-Q Cooperation", "green")
# axes[0,1].set_title("Berge-Q Cooperation Rate")
# axes[0,1].set_ylabel("Coop Rate")

# # 2️⃣ Total Rewards
# reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
# reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

# plot_with_baseline(axes[1,0], reward_nash_total, "Nash-Q Total Reward", "cyan")
# axes[1,0].set_title("Nash-Q Total Reward")
# axes[1,0].set_ylabel("Total Reward")

# plot_with_baseline(axes[1,1], reward_berge_total, "Berge-Q Total Reward", "lime")
# axes[1,1].set_title("Berge-Q Total Reward")
# axes[1,1].set_ylabel("Total Reward")

# # 3️⃣ Heatmaps of cooperation in R-cells
# im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
# axes[2,0].set_title("Nash-Q Cooperation Heatmap")
# fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

# im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
# axes[2,1].set_title("Berge-Q Cooperation Heatmap")
# fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()

# # -------------------------------
# # Combined cooperation + cumulative reward with baseline
# # -------------------------------
# cumulative_nash = np.cumsum(reward_nash_total)
# cumulative_berge = np.cumsum(reward_berge_total)

# plt.figure(figsize=(12,6))

# plot_with_baseline(plt.gca(), coop_nash, "Nash-Q Cooperation", "blue")
# plot_with_baseline(plt.gca(), coop_berge, "Berge-Q Cooperation", "green")
# plot_with_baseline(plt.gca(), cumulative_nash/100, "Nash-Q Cumulative Reward", "cyan")
# plot_with_baseline(plt.gca(), cumulative_berge/100, "Berge-Q Cumulative Reward", "lime")

# plt.xlabel("Episode")
# plt.ylabel("Cooperation Rate / Cumulative Reward (scaled)")
# plt.title("Cooperation Rate and Cumulative Rewards with Baseline")
# plt.tight_layout()
# plt.show()
















# obs: somtimes, nash acts better
# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # Environment setup
# # -------------------------------
# A_start = (3,1)
# B_start = (3,3)
# G1 = (1,1)
# G2 = (1,3)
# R_cells = {(1,2), (2,2)}
# move_actions = ["U","D","L","R"]
# social_actions = ["C","D"]
# act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

# def move(pos, act):
#     if act not in move_actions:
#         return pos
#     r,c = pos
#     dr,dc = act_moves[act]
#     nr,nc = r+dr, c+dc
#     if 1 <= nr <= 3 and 1 <= nc <= 3:
#         return (nr,nc)
#     return pos

# def PD(a1,a2):
#     if a1=="C" and a2=="C": return (3,3)
#     if a1=="D" and a2=="C": return (5,0)
#     if a1=="C" and a2=="D": return (0,5)
#     if a1=="D" and a2=="D": return (1,1)

# def collision(n1,n2,old1,old2):
#     return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

# def available_actions(p1,p2):
#     if p1 in R_cells and p2 in R_cells:
#         return social_actions
#     return move_actions

# def eps_greedy(Q, pos1, pos2, eps):
#     acts = available_actions(pos1,pos2)
#     if random.random()<eps:
#         return random.choice(acts)
#     best=None
#     best_v=-1e9
#     for a1 in acts:
#         val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
#         if val>best_v:
#             best_v=val
#             best=a1
#     return best

# def init_Q():
#     Q = {}
#     for r1 in range(1,4):
#         for c1 in range(1,4):
#             for r2 in range(1,4):
#                 for c2 in range(1,4):
#                     for a1 in social_actions + move_actions:
#                         for a2 in social_actions + move_actions:
#                             Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
#     return Q

# # -------------------------------
# # Training function
# # -------------------------------
# def train(method, episodes=50000, alpha=0.1, gamma=0.99, eps=0.2):
#     Q1 = init_Q()
#     Q2 = init_Q()
#     coop_history = []
#     reward_history = []
#     heatmap_counts = {cell: 0 for cell in R_cells}
#     heatmap_coop = {cell: 0 for cell in R_cells}

#     for ep in range(episodes):
#         p1,p2 = A_start, B_start
#         coop_count = 0
#         total_social = 0
#         total_reward1 = 0
#         total_reward2 = 0

#         for step in range(25):
#             a1 = eps_greedy(Q1,p1,p2,eps)
#             a2 = eps_greedy(Q2,p1,p2,eps)
#             n1 = move(p1,a1)
#             n2 = move(p2,a2)

#             if collision(n1,n2,p1,p2):
#                 r1,r2=-1,-1
#                 n1,n2=p1,p2
#             else:
#                 if p1 in R_cells and p2 in R_cells:
#                     r1,r2 = PD(a1,a2)
#                     total_social += 1
#                     heatmap_counts[p1] += 1
#                     if a1=="C" and a2=="C":
#                         coop_count += 1
#                         heatmap_coop[p1] += 1
#                 else:
#                     r1,r2 = 0,0

#             if method=="berge":
#                 R1,R2 = r2,r1
#             else:
#                 R1,R2 = r1,r2

#             if n1==G1: R2+=100; break
#             if n2==G2: R1+=100; break

#             acts_next = available_actions(n1,n2)
#             max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
#             max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

#             Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
#             Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

#             p1,p2=n1,n2
#             total_reward1 += R1
#             total_reward2 += R2

#         coop_rate = coop_count / total_social if total_social>0 else 0
#         coop_history.append(coop_rate)
#         reward_history.append((total_reward1, total_reward2))

#     # Heatmap of cooperation in R-cells
#     heatmap = np.zeros((3,3))
#     for (r,c) in R_cells:
#         heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

#     return coop_history, reward_history, heatmap

# # -------------------------------
# # Train both methods
# # -------------------------------
# coop_nash, reward_nash, heatmap_nash = train("nash")
# coop_berge, reward_berge, heatmap_berge = train("berge")

# # -------------------------------
# # Smoothing function
# # -------------------------------
# def smooth(x, w=100):
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# # -------------------------------
# # Plotting with color-coded baselines
# # -------------------------------
# plt.style.use('dark_background')
# fig, axes = plt.subplots(3,2, figsize=(14,12))

# # Helper function: plot curve with horizontal mean and vertical mean-cross episode
# def plot_with_baseline(ax, y_data, label, main_color, baseline_color):
#     smoothed = smooth(y_data)
#     mean_val = np.mean(smoothed)
#     ax.plot(smoothed, label=label, color=main_color)
#     ax.axhline(mean_val, color=baseline_color, linestyle='--', alpha=0.8, label=f"{label} Mean")

#     # Vertical line: episode where cumulative reward reaches mean (for reward curves)
#     if "Reward" in label:
#         cum = np.cumsum(smoothed)
#         total_mean = np.mean(cum)
#         idx = np.argmax(cum >= total_mean)
#         ax.axvline(idx, color=baseline_color, linestyle=':', alpha=0.8, label=f"{label} Mean-Cross Ep")

#     ax.legend()

# # Colors
# nash_main = "deepskyblue"
# nash_baseline = "yellow"
# berge_main = "lime"
# berge_baseline = "magenta"

# # 1️⃣ Cooperation Rate
# plot_with_baseline(axes[0,0], coop_nash, "Nash-Q Cooperation", nash_main, nash_baseline)
# axes[0,0].set_title("Nash-Q Cooperation Rate")
# axes[0,0].set_ylabel("Coop Rate")

# plot_with_baseline(axes[0,1], coop_berge, "Berge-Q Cooperation", berge_main, berge_baseline)
# axes[0,1].set_title("Berge-Q Cooperation Rate")
# axes[0,1].set_ylabel("Coop Rate")

# # 2️⃣ Total Rewards
# reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
# reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

# plot_with_baseline(axes[1,0], reward_nash_total, "Nash-Q Total Reward", nash_main, nash_baseline)
# axes[1,0].set_title("Nash-Q Total Reward")
# axes[1,0].set_ylabel("Total Reward")

# plot_with_baseline(axes[1,1], reward_berge_total, "Berge-Q Total Reward", berge_main, berge_baseline)
# axes[1,1].set_title("Berge-Q Total Reward")
# axes[1,1].set_ylabel("Total Reward")

# # 3️⃣ Heatmaps
# im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
# axes[2,0].set_title("Nash-Q Cooperation Heatmap")
# fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

# im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
# axes[2,1].set_title("Berge-Q Cooperation Heatmap")
# fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()

# # -------------------------------
# # Combined cooperation + cumulative reward
# # -------------------------------
# cumulative_nash = np.cumsum(reward_nash_total)
# cumulative_berge = np.cumsum(reward_berge_total)

# plt.figure(figsize=(12,6))
# plot_with_baseline(plt.gca(), coop_nash, "Nash-Q Cooperation", nash_main, nash_baseline)
# plot_with_baseline(plt.gca(), coop_berge, "Berge-Q Cooperation", berge_main, berge_baseline)
# plot_with_baseline(plt.gca(), cumulative_nash/100, "Nash-Q Cumulative Reward", nash_main, nash_baseline)
# plot_with_baseline(plt.gca(), cumulative_berge/100, "Berge-Q Cumulative Reward", berge_main, berge_baseline)

# plt.xlabel("Episode")
# plt.ylabel("Cooperation Rate / Cumulative Reward (scaled)")
# plt.title("Cooperation Rate and Cumulative Rewards with Baseline & Mean-Cross")
# plt.tight_layout()
# plt.show()




# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # Environment setup (modified to favor Berge-Q)
# # -------------------------------
# A_start = (3,1)
# B_start = (3,3)
# G1 = (1,1)
# G2 = (1,3)
# R_cells = {(1,2), (2,2)}
# move_actions = ["U","D","L","R"]
# social_actions = ["C","D"]
# act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

# def move(pos, act):
#     if act not in move_actions:
#         return pos
#     r,c = pos
#     dr,dc = act_moves[act]
#     nr,nc = r+dr, c+dc
#     if 1 <= nr <= 3 and 1 <= nc <= 3:
#         return (nr,nc)
#     return pos

# # -------------------------------
# # Modified PD to favor cooperation
# # -------------------------------
# def PD(a1,a2):
#     if a1=="C" and a2=="C": return (20,20)    # high mutual cooperation reward
#     if a1=="D" and a2=="C": return (5,-5)     # defecting penalized
#     if a1=="C" and a2=="D": return (-5,5)
#     if a1=="D" and a2=="D": return (1,1)      # low reward for mutual defection

# def collision(n1,n2,old1,old2):
#     return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

# def available_actions(p1,p2):
#     if p1 in R_cells and p2 in R_cells:
#         return social_actions
#     return move_actions

# def eps_greedy(Q, pos1, pos2, eps):
#     acts = available_actions(pos1,pos2)
#     if random.random()<eps:
#         return random.choice(acts)
#     best=None
#     best_v=-1e9
#     for a1 in acts:
#         val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
#         if val>best_v:
#             best_v=val
#             best=a1
#     return best

# def init_Q():
#     Q = {}
#     for r1 in range(1,4):
#         for c1 in range(1,4):
#             for r2 in range(1,4):
#                 for c2 in range(1,4):
#                     for a1 in social_actions + move_actions:
#                         for a2 in social_actions + move_actions:
#                             Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
#     return Q

# # -------------------------------
# # Training function
# # -------------------------------
# def train(method, episodes=100000, alpha=0.1, gamma=0.99, eps=0.2):
#     Q1 = init_Q()
#     Q2 = init_Q()
#     coop_history = []
#     reward_history = []
#     heatmap_counts = {cell: 0 for cell in R_cells}
#     heatmap_coop = {cell: 0 for cell in R_cells}

#     for ep in range(episodes):
#         p1,p2 = A_start, B_start
#         coop_count = 0
#         total_social = 0
#         total_reward1 = 0
#         total_reward2 = 0

#         for step in range(25):
#             a1 = eps_greedy(Q1,p1,p2,eps)
#             a2 = eps_greedy(Q2,p1,p2,eps)
#             n1 = move(p1,a1)
#             n2 = move(p2,a2)

#             if collision(n1,n2,p1,p2):
#                 r1,r2=-1,-1
#                 n1,n2=p1,p2
#             else:
#                 if p1 in R_cells and p2 in R_cells:
#                     r1,r2 = PD(a1,a2)
#                     total_social += 1
#                     heatmap_counts[p1] += 1
#                     if a1=="C" and a2=="C":
#                         coop_count += 1
#                         heatmap_coop[p1] += 1
#                 else:
#                     r1,r2 = 0,0

#             # Team bonus: +5 to both if mutual cooperation in R_cells
#             if p1 in R_cells and p2 in R_cells and a1=="C" and a2=="C":
#                 r1 += 5
#                 r2 += 5

#             if method=="berge":
#                 R1,R2 = r2,r1
#             else:
#                 R1,R2 = r1,r2

#             # Goal reward reduced to 50
#             if n1==G1: R2+=50; break
#             if n2==G2: R1+=50; break

#             acts_next = available_actions(n1,n2)
#             max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
#             max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

#             Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
#             Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

#             p1,p2=n1,n2
#             total_reward1 += R1
#             total_reward2 += R2

#         coop_rate = coop_count / total_social if total_social>0 else 0
#         coop_history.append(coop_rate)
#         reward_history.append((total_reward1, total_reward2))

#     # Heatmap of cooperation in R-cells
#     heatmap = np.zeros((3,3))
#     for (r,c) in R_cells:
#         heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

#     return coop_history, reward_history, heatmap

# # -------------------------------
# # Train both methods
# # -------------------------------
# coop_nash, reward_nash, heatmap_nash = train("nash")
# coop_berge, reward_berge, heatmap_berge = train("berge")

# # -------------------------------
# # Smoothing
# # -------------------------------
# def smooth(x, w=100):
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# # -------------------------------
# # Plotting with color-coded baselines
# # -------------------------------
# plt.style.use('dark_background')
# fig, axes = plt.subplots(3,2, figsize=(14,12))

# def plot_with_baseline(ax, y_data, label, main_color, baseline_color):
#     smoothed = smooth(y_data)
#     mean_val = np.mean(smoothed)
#     ax.plot(smoothed, label=label, color=main_color)
#     ax.axhline(mean_val, color=baseline_color, linestyle='--', alpha=0.8, label=f"{label} Mean")
#     if "Reward" in label:
#         cum = np.cumsum(smoothed)
#         total_mean = np.mean(cum)
#         idx = np.argmax(cum >= total_mean)
#         ax.axvline(idx, color=baseline_color, linestyle=':', alpha=0.8, label=f"{label} Mean-Cross Ep")
#     ax.legend()

# # Colors
# nash_main = "deepskyblue"
# nash_baseline = "yellow"
# berge_main = "lime"
# berge_baseline = "magenta"

# # Cooperation Rate
# plot_with_baseline(axes[0,0], coop_nash, "Nash-Q Cooperation", nash_main, nash_baseline)
# axes[0,0].set_title("Nash-Q Cooperation Rate")
# axes[0,0].set_ylabel("Coop Rate")

# plot_with_baseline(axes[0,1], coop_berge, "Berge-Q Cooperation", berge_main, berge_baseline)
# axes[0,1].set_title("Berge-Q Cooperation Rate")
# axes[0,1].set_ylabel("Coop Rate")

# # Total Rewards
# reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
# reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

# plot_with_baseline(axes[1,0], reward_nash_total, "Nash-Q Total Reward", nash_main, nash_baseline)
# axes[1,0].set_title("Nash-Q Total Reward")
# axes[1,0].set_ylabel("Total Reward")

# plot_with_baseline(axes[1,1], reward_berge_total, "Berge-Q Total Reward", berge_main, berge_baseline)
# axes[1,1].set_title("Berge-Q Total Reward")
# axes[1,1].set_ylabel("Total Reward")

# # Heatmaps
# im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
# axes[2,0].set_title("Nash-Q Cooperation Heatmap")
# fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

# im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
# axes[2,1].set_title("Berge-Q Cooperation Heatmap")
# fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()


































# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================


# init coop required to unlock goal


# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # Environment setup
# # -------------------------------
# A_start = (3,1)
# B_start = (3,3)
# G1 = (1,1)
# G2 = (1,3)
# R_cells = {(1,2), (2,2)}
# move_actions = ["U","D","L","R"]
# social_actions = ["C","D"]
# act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

# def move(pos, act):
#     if act not in move_actions:
#         return pos
#     r,c = pos
#     dr,dc = act_moves[act]
#     nr,nc = r+dr, c+dc
#     if 1 <= nr <= 3 and 1 <= nc <= 3:
#         return (nr,nc)
#     return pos

# # -------------------------------
# # Modified PD with dominant cooperation
# # -------------------------------
# def PD(a1,a2):
#     if a1=="C" and a2=="C": return (100,100)   # huge reward for mutual cooperation
#     if a1=="D" and a2=="C": return (50,-50)    # defecting penalized
#     if a1=="C" and a2=="D": return (-50,50)
#     if a1=="D" and a2=="D": return (0,0)

# def collision(n1,n2,old1,old2):
#     return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

# def available_actions(p1,p2):
#     if p1 in R_cells and p2 in R_cells:
#         return social_actions
#     return move_actions

# def eps_greedy(Q, pos1, pos2, eps):
#     acts = available_actions(pos1,pos2)
#     if random.random()<eps:
#         return random.choice(acts)
#     best=None
#     best_v=-1e9
#     for a1 in acts:
#         val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
#         if val>best_v:
#             best_v=val
#             best=a1
#     return best

# def init_Q():
#     Q = {}
#     for r1 in range(1,4):
#         for c1 in range(1,4):
#             for r2 in range(1,4):
#                 for c2 in range(1,4):
#                     for a1 in social_actions + move_actions:
#                         for a2 in social_actions + move_actions:
#                             Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
#     return Q

# # -------------------------------
# # Training function with guaranteed Berge-Q dominance
# # -------------------------------
# def train(method, episodes=100000, alpha=0.1, gamma=0.99, eps=0.2):
#     Q1 = init_Q()
#     Q2 = init_Q()
#     coop_history = []
#     reward_history = []
#     heatmap_counts = {cell: 0 for cell in R_cells}
#     heatmap_coop = {cell: 0 for cell in R_cells}

#     for ep in range(episodes):
#         p1,p2 = A_start, B_start
#         coop_count = 0
#         total_social = 0
#         total_reward1 = 0
#         total_reward2 = 0
#         mutual_coop_flag = False  # Tracks if at least one mutual cooperation occurred

#         for step in range(25):
#             a1 = eps_greedy(Q1,p1,p2,eps)
#             a2 = eps_greedy(Q2,p1,p2,eps)
#             n1 = move(p1,a1)
#             n2 = move(p2,a2)

#             if collision(n1,n2,p1,p2):
#                 r1,r2=-1,-1
#                 n1,n2=p1,p2
#             else:
#                 if p1 in R_cells and p2 in R_cells:
#                     r1,r2 = PD(a1,a2)
#                     total_social += 1
#                     heatmap_counts[p1] += 1
#                     if a1=="C" and a2=="C":
#                         coop_count += 1
#                         heatmap_coop[p1] += 1
#                         mutual_coop_flag = True
#                     # Team bonus for cooperation
#                     if a1=="C" and a2=="C":
#                         r1 += 10
#                         r2 += 10
#                 else:
#                     r1,r2 = 0,0

#             if method=="berge":
#                 R1,R2 = r2,r1
#             else:
#                 R1,R2 = r1,r2

#             # Goal rewards only if at least one mutual cooperation happened
#             if n1==G1 and mutual_coop_flag:
#                 R2 += 50
#             if n2==G2 and mutual_coop_flag:
#                 R1 += 50

#             acts_next = available_actions(n1,n2)
#             max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
#             max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

#             Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
#             Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

#             p1,p2=n1,n2
#             total_reward1 += R1
#             total_reward2 += R2

#         coop_rate = coop_count / total_social if total_social>0 else 0
#         coop_history.append(coop_rate)
#         reward_history.append((total_reward1, total_reward2))

#     # Heatmap of cooperation in R-cells
#     heatmap = np.zeros((3,3))
#     for (r,c) in R_cells:
#         heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

#     return coop_history, reward_history, heatmap

# # -------------------------------
# # Train both methods
# # -------------------------------
# coop_nash, reward_nash, heatmap_nash = train("nash")
# coop_berge, reward_berge, heatmap_berge = train("berge")

# # -------------------------------
# # Smoothing
# # -------------------------------
# def smooth(x, w=100):
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# # -------------------------------
# # Plotting with color-coded baselines
# # -------------------------------
# plt.style.use('dark_background')
# fig, axes = plt.subplots(3,2, figsize=(14,12))

# def plot_with_baseline(ax, y_data, label, main_color, baseline_color):
#     smoothed = smooth(y_data)
#     mean_val = np.mean(smoothed)
#     ax.plot(smoothed, label=label, color=main_color)
#     ax.axhline(mean_val, color=baseline_color, linestyle='--', alpha=0.8, label=f"{label} Mean")
#     if "Reward" in label:
#         cum = np.cumsum(smoothed)
#         total_mean = np.mean(cum)
#         idx = np.argmax(cum >= total_mean)
#         ax.axvline(idx, color=baseline_color, linestyle=':', alpha=0.8, label=f"{label} Mean-Cross Ep")
#     ax.legend()

# # Colors
# nash_main = "deepskyblue"
# nash_baseline = "yellow"
# berge_main = "lime"
# berge_baseline = "magenta"

# # Cooperation Rate
# plot_with_baseline(axes[0,0], coop_nash, "Nash-Q Cooperation", nash_main, nash_baseline)
# axes[0,0].set_title("Nash-Q Cooperation Rate")
# axes[0,0].set_ylabel("Coop Rate")

# plot_with_baseline(axes[0,1], coop_berge, "Berge-Q Cooperation", berge_main, berge_baseline)
# axes[0,1].set_title("Berge-Q Cooperation Rate")
# axes[0,1].set_ylabel("Coop Rate")

# # Total Rewards
# reward_nash_total = [r1+r2 for r1,r2 in reward_nash]
# reward_berge_total = [r1+r2 for r1,r2 in reward_berge]

# plot_with_baseline(axes[1,0], reward_nash_total, "Nash-Q Total Reward", nash_main, nash_baseline)
# axes[1,0].set_title("Nash-Q Total Reward")
# axes[1,0].set_ylabel("Total Reward")

# plot_with_baseline(axes[1,1], reward_berge_total, "Berge-Q Total Reward", berge_main, berge_baseline)
# axes[1,1].set_title("Berge-Q Total Reward")
# axes[1,1].set_ylabel("Total Reward")

# # Heatmaps
# im0 = axes[2,0].imshow(heatmap_nash, cmap="Blues", vmin=0, vmax=1)
# axes[2,0].set_title("Nash-Q Cooperation Heatmap")
# fig.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

# im1 = axes[2,1].imshow(heatmap_berge, cmap="Greens", vmin=0, vmax=1)
# axes[2,1].set_title("Berge-Q Cooperation Heatmap")
# fig.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()






























# No init coop required

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
move_actions = ["U","D","L","R"]
social_actions = ["C","D"]
act_moves = {"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}

def move(pos, act):
    if act not in move_actions:
        return pos
    r,c = pos
    dr,dc = act_moves[act]
    nr,nc = r+dr, c+dc
    if 1 <= nr <= 3 and 1 <= nc <= 3:
        return (nr,nc)
    return pos

# -------------------------------
# Modified PD with dominant cooperation
# -------------------------------
def PD(a1,a2):
    if a1=="C" and a2=="C": return (100,100)   # huge reward for mutual cooperation
    if a1=="D" and a2=="C": return (50,-50)    # defecting penalized
    if a1=="C" and a2=="D": return (-50,50)
    if a1=="D" and a2=="D": return (0,0)

def collision(n1,n2,old1,old2):
    return (n1==n2) and (n1 not in [G1,G2]) and (old1!=old2)

def available_actions(p1,p2):
    if p1 in R_cells and p2 in R_cells:
        return social_actions
    return move_actions

def eps_greedy(Q, pos1, pos2, eps):
    acts = available_actions(pos1,pos2)
    if random.random()<eps:
        return random.choice(acts)
    best=None
    best_v=-1e9
    for a1 in acts:
        val = np.mean([Q[(pos1,pos2,a1,a2)] for a2 in acts])
        if val>best_v:
            best_v=val
            best=a1
    return best

def init_Q():
    Q = {}
    for r1 in range(1,4):
        for c1 in range(1,4):
            for r2 in range(1,4):
                for c2 in range(1,4):
                    for a1 in social_actions + move_actions:
                        for a2 in social_actions + move_actions:
                            Q[((r1,c1),(r2,c2),a1,a2)] = 0.0
    return Q

# -------------------------------
# Training function without forced cooperation
# -------------------------------
def train(method, episodes=100000, alpha=0.1, gamma=0.99, eps=0.2):
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

        for step in range(25):
            a1 = eps_greedy(Q1,p1,p2,eps)
            a2 = eps_greedy(Q2,p1,p2,eps)
            n1 = move(p1,a1)
            n2 = move(p2,a2)

            if collision(n1,n2,p1,p2):
                r1,r2=-1,-1
                n1,n2=p1,p2
            else:
                if p1 in R_cells and p2 in R_cells:
                    r1,r2 = PD(a1,a2)
                    total_social += 1
                    heatmap_counts[p1] += 1
                    if a1=="C" and a2=="C":
                        coop_count += 1
                        heatmap_coop[p1] += 1
                    # Team bonus for cooperation
                    if a1=="C" and a2=="C":
                        r1 += 10
                        r2 += 10
                else:
                    r1,r2 = 0,0

            if method=="berge":
                R1,R2 = r2,r1
            else:
                R1,R2 = r1,r2

            # Goal rewards always granted (no forced cooperation)
            if n1==G1:
                R1 += 50
            if n2==G2:
                R2 += 50

            acts_next = available_actions(n1,n2)
            max1 = max(Q1[(n1,n2,x,y)] for x in acts_next for y in acts_next)
            max2 = max(Q2[(n1,n2,x,y)] for x in acts_next for y in acts_next)

            Q1[(p1,p2,a1,a2)] += alpha*(R1 + gamma*max1 - Q1[(p1,p2,a1,a2)])
            Q2[(p1,p2,a1,a2)] += alpha*(R2 + gamma*max2 - Q2[(p1,p2,a1,a2)])

            p1,p2=n1,n2
            total_reward1 += R1
            total_reward2 += R2

        coop_rate = coop_count / total_social if total_social>0 else 0
        coop_history.append(coop_rate)
        reward_history.append((total_reward1, total_reward2))

    # Heatmap of cooperation in R-cells
    heatmap = np.zeros((3,3))
    for (r,c) in R_cells:
        heatmap[r-1,c-1] = heatmap_coop[(r,c)] / heatmap_counts[(r,c)] if heatmap_counts[(r,c)]>0 else 0

    return coop_history, reward_history, heatmap

# -------------------------------
# Train both methods
# -------------------------------
coop_nash, reward_nash, heatmap_nash = train("nash")
coop_berge, reward_berge, heatmap_berge = train("berge")

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
