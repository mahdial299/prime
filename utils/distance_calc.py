


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dist_to_reward_cell(pos):
    return min(manhattan(pos, pd_cell) for pd_cell in PD_CELLS)
