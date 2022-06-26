import copy
from copy import deepcopy
from multiprocessing import pool
import numpy as np

index_to_action = ["UP", "DOWN", "RIGHT", "LEFT"]

def get_wall_locations(mdp):
    walls = []
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if is_wall(mdp, row, col):
                walls.append(row * mdp.num_col + col)
    return walls

def is_wall(mdp, row, col):
    return mdp.board[row][col] == "WALL"


def is_terminal(mdp, row, col):
    return (row, col) in mdp.terminal_states


def get_util_times_p_for_state(mdp, U, row, col, action):
    util = 0.0
    for i, a in enumerate(mdp.transition_function):
        r, c = mdp.step((row, col), a)
        util += mdp.transition_function[action][i] * U[r][c]
    return util


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    gamma = mdp.gamma
    stop_delta = (epsilon * (1 - gamma)) / gamma
    U_r = copy.deepcopy(U_init)
    U_tag = copy.deepcopy(U_init)
    while True:
        U_r = copy.deepcopy(U_tag)
        delta = 0
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if is_wall(mdp, row, col):
                    continue
                cur_state = (row, col)
                cur_reward = float(mdp.board[row][col])
                if is_terminal(mdp, row, col):
                    U_tag[row][col] = cur_reward
                    continue
                best_action = 0.0
                for action in mdp.transition_function:
                    cur_action = 0
                    for i, a in enumerate(mdp.actions):
                        next_state = mdp.step(cur_state, a)
                        cur_action += mdp.transition_function[action][i] * U_r[next_state[0]][next_state[1]]
                    if cur_action > best_action:
                        best_action = cur_action
                U_tag[row][col] = cur_reward + gamma * best_action
                if abs(U_tag[row][col] - U_r[row][col]) > delta:
                    delta = abs(U_tag[row][col] - U_r[row][col])
        if gamma == 1:
            if delta == 0:
                break
        elif delta < stop_delta:
            break
    return U_r


def get_policy(mdp, U):
    ret_policy = [[0] * mdp.num_col for i in range(mdp.num_row)]
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            cur_state = (row, col)
            all_results = []
            for action in mdp.actions:
                cur_sum = 0
                for i, loc_action in enumerate(mdp.actions):
                    next_state = mdp.step(cur_state, loc_action)
                    loc_action_prob = mdp.transition_function[action][i]
                    cur_sum += loc_action_prob * U[next_state[0]][next_state[1]]
                all_results.append(cur_sum)
            best_result = max(all_results)
            ret_policy[row][col] = index_to_action[all_results.index(
                best_result)]
    return ret_policy


def policy_evaluation(mdp, policy):
    n_rows = mdp.num_row
    n_cols = mdp.num_col
    wall_locations = get_wall_locations(mdp)
    P_mat = np.zeros((int(n_rows * n_cols), int(n_rows * n_cols)))
    for row in range(n_rows):
        for col in range(n_cols):
            if is_wall(mdp, row, col): # wall is not a state so no row in P_mat
                continue
            if is_terminal(mdp, row, col): # row full of zeroes
                continue
            cur_state = (row, col)
            for i, action in enumerate(mdp.actions):
                (nex_row, nex_col) = mdp.step(cur_state, action)
                P_mat[row * n_cols + col][nex_row * n_cols + nex_col] += mdp.transition_function[policy[row][col]][i]
    
    for wall in wall_locations:
        P_mat = np.delete(P_mat, wall, 0) # delete wall row
        P_mat = np.delete(P_mat, wall, 1) # delete wall col
    reward_vec = []
    for row in range(n_rows):
        for col in range(n_cols):
            if is_wall(mdp, row, col):
                continue
            cur_reward = float(mdp.board[row][col])
            reward_vec.append(cur_reward)
    id_mat = np.identity(len(reward_vec)) # equations from tutorial.
    g_P_mat = P_mat.dot(mdp.gamma)
    inv_mat = np.linalg.inv(np.subtract(id_mat, g_P_mat))
    U_p = inv_mat.dot(reward_vec)

    n_wall = 0
    U_r = np.zeros((n_rows, n_cols))
    for row in range(n_rows):
        for col in range(n_cols):
            if is_wall(mdp, row, col):
                n_wall += 1
                U_r[row][col] = 0
            else:
                U_r[row][col] = U_p[row * n_cols + col - n_wall]
    return U_r


def policy_iteration(mdp, policy_init):
    ret_policy = copy.deepcopy(policy_init)
    while True:
        un_changed = True
        U_r = policy_evaluation(mdp, ret_policy)
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if is_wall(mdp, row, col) or is_terminal(mdp, row, col):
                    continue
                max_util = np.NINF
                best_action = None
                for action in mdp.actions:
                    cur_util = get_util_times_p_for_state(mdp, U_r, row, col, action) # P(s'| s, a) * U[s']
                    if cur_util > max_util:
                        best_action = action
                        max_util = cur_util
                cur_action = ret_policy[row][col]
                if best_action != cur_action:
                    ret_policy[row][col] = best_action # getting argmax
                    un_changed = False
        if un_changed:
            break
    return ret_policy
