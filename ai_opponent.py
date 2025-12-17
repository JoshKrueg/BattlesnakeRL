"""
Module: AI Opponent Implementations
Purpose: Various AI algorithms including heuristic, MCTS, and alpha-beta search
Optimized: Numba-accelerated flood fill and game analysis
"""

from game_state import GameState
from game_state_optimized import *
import torch
import numpy as np
from numba import njit, jit

DIRECTIONS_TUPLE = ((1, 0), (0, 1), (-1, 0), (0, -1))

@njit
def flood_fill(s0_territory, s1_territory, food_array, blocked_mask, snake0_length, snake1_length):
    t0, t1, d0, d1, f0, f1 = 0, 0, -1, -1, 0, 0

    for i in range(1, 19):
        s0_expanded = (
            s0_territory[0:19, 1:20] |
            s0_territory[1:20, 0:19] |
            s0_territory[1:20, 2:21] |
            s0_territory[2:21, 1:20]
        )
        s1_expanded = (
            s1_territory[0:19, 1:20] |
            s1_territory[1:20, 0:19] |
            s1_territory[1:20, 2:21] |
            s1_territory[2:21, 1:20]
        )

        open = ~blocked_mask

        if snake0_length > snake1_length:
            s0_new_territory = s0_expanded & open & ~s1_territory[1:20, 1:20]
            s1_new_territory = s1_expanded & open & ~(s0_expanded | s1_territory[1:20, 1:20])
        else:
            s0_new_territory = s0_expanded & open & ~(s1_expanded | s1_territory[1:20, 1:20])
            s1_new_territory = s1_expanded & open & ~s0_territory[1:20, 1:20]

        s0_frontier = s0_new_territory & ~s0_territory[1:20, 1:20]
        s1_frontier = s1_new_territory & ~s1_territory[1:20, 1:20]

        if d0 == -1 and not s0_frontier.any():
            d0 = i
        if d1 == -1 and not s1_frontier.any():
            d1 = i

        if f0 == 0 and np.any(s0_new_territory & food_array):
            f0 = i
        if f1 == 0 and np.any(s1_new_territory & food_array):
            f1 = i

        s0_territory[1:20, 1:20] |= s0_frontier
        s1_territory[1:20, 1:20] |= s1_frontier

    if d0 == -1:
        d0 = 19
    if d1 == -1:
        d1 = 19
    if f0 == 0:
        f0 = 19
    if f1 == 0:
        f1 = 19

    t0 = np.sum(s0_territory)
    t1 = np.sum(s1_territory)

    return t0, t1, d0, d1, f0, f1

@njit
def flood_fill_heuristic(state, move_result, k1, k2, k3, k4, k5):
    (gameboard, snake0_coords, snake0_length, snake0_health, snake1_coords, snake1_length, snake1_health) = state

    if move_result == -1:
        return -np.inf
    elif move_result == 1:
        return np.inf

    t0,t1,d0,d1,f0,f1 = flood_fill(
        gameboard[3] != 0,
        gameboard[5] != 0,
        gameboard[1][1:20, 1:20] != 0,
        t_blocked_mask(gameboard)[1:20, 1:20] != 0,
        snake0_length,
        snake1_length
    )

    mult, fs = -1, f0
    
    foodscore = 0 
    fmult = 0
    fminscaler = k2*(snake0_health / 100) + k3*((100-snake0_health)/100) #hunger weighted averages
    fselfscaler = k4*(snake0_health / 100) + k5*((100-snake0_health)/100)
    
    if f1 < f0: fmult = 1
    elif f1 > f0: fmult = -1

    if f0 == f1:
        foodscore = 0
    else:
        foodscore = fminscaler*fmult*(19-min(f0,f1))/19 + fselfscaler*mult*(19-fs)/38
    out = (t1-t0)/361 + k1*(d1-d0)/38 + foodscore #the additional factors of 361 and 38 are used to normalize the results before
    #multiplying by the coefficients so that coefficients of similar values convey simlar importances
    return mult*out

@njit
def negamax(state: tuple, r, depth: int, alpha: float, beta: float, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3):
    if r == -1:
        return -np.inf
    elif r == 1:
        return np.inf
    if depth <= 0: 
        return flood_fill_heuristic(state, r, k1, k2, k3, k4, k5)

    state = t_flip_player(state)
    valid_moves = t_get_valid_moves(state)
    score = -np.inf
    for move in valid_moves:
        next_state, r = t_get_next_state(state, move)            
        score = max(score, -negamax(next_state, r, depth - 1, -beta, -alpha, k1, k2, k3, k4, k5))
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return score

@njit(fastmath=True)
def get_manhattan_dist(p1_r, p1_c, p2_r, p2_c):
    return abs(p1_r - p2_r) + abs(p1_c - p2_c)

@njit(fastmath=True)
def get_best_greedy_move(state, move_mask):
    """
    Optimized greedy move selector. 
    Scans the board ONCE to find food, avoiding np.argwhere allocation.
    """
    # Unpack for speed
    board_food = state[0][1] # Assuming channel 1 is food
    head_pos = state[1]
    my_r, my_c = head_pos[0], head_pos[1]
    
    H, W = board_food.shape
    
    # 1. Find Targets (Food) manually to avoid allocating an array
    # We use a fixed-size buffer on the stack. Max 20 food items is plenty for Snake.
    target_buffer = np.empty((20, 2), dtype=np.int32)
    target_count = 0
    
    # Fast Board Scan
    for r in range(H):
        for c in range(W):
            if board_food[r, c] == 1:
                target_buffer[target_count, 0] = r
                target_buffer[target_count, 1] = c
                target_count += 1
                if target_count >= 20: break 
        if target_count >= 20: break

    # Fallback to Tail if no food (Channel 2 is usually bodies)
    if target_count == 0:
        board_bodies = state[0][2]
        for r in range(H):
            for c in range(W):
                if board_bodies[r, c] == 1: # Assuming 1 is tail/body
                    target_buffer[target_count, 0] = r
                    target_buffer[target_count, 1] = c
                    target_count += 1
                    if target_count >= 20: break
            if target_count >= 20: break
            
    # If still no targets, return safe default
    if target_count == 0:
        # Just pick the first valid move found
        for i in range(4):
            if move_mask[i]: return i
        return 0

    # 2. Find Closest Target
    best_move_idx = -1
    min_global_dist = 999999
    
    # Iterate moves: Down(0), Right(1), Up(2), Left(3)
    # Using enumerate on tuple is unrolled by Numba
    for i, move_dir in enumerate(DIRECTIONS_TUPLE):
        if not move_mask[i]:
            continue
            
        # Calculate new head position
        # Note: Logic assumes (y, x) or (x, y). Standard matrix is (row, col)
        # Adjust indices [0]/[1] based on your specific coordinate system if needed.
        # Assuming: move_dir is (row_delta, col_delta)
        next_r = my_r + move_dir[0]
        next_c = my_c + move_dir[1]
        
        # Check dist to all targets
        local_min = 999999
        for t in range(target_count):
            t_r = target_buffer[t, 0]
            t_c = target_buffer[t, 1]
            dist = abs(t_r - next_r) + abs(t_c - next_c)
            if dist < local_min:
                local_min = dist
        
        if local_min < min_global_dist:
            min_global_dist = local_min
            best_move_idx = i
            
            # Optimization: Can't get better than 0 distance (on top of food)
            if min_global_dist == 0:
                return best_move_idx

    if best_move_idx == -1:
        # Fallback if logic failed (e.g. valid moves existed but logic skipped)
        for i in range(4):
            if move_mask[i]: return i
        return 0
        
    return best_move_idx

@njit(types.Tuple((GameStateType, types.int16, types.boolean, types.int32))(GameStateType, types.Array(types.float32, 2, 'C'), types.Array(types.int32, 2, 'C'), types.float32), fastmath=True)
def node_select(root_state, stats, children, c):
    node = 0
    state = root_state
    r = 0
    f = False
    
    # Pre-allocate array for Q+U calc to avoid allocating inside loop?
    # Numba handles small array math well, but avoiding creation is better.
    # However, 'children' index is non-contiguous, so standard slice is fine.
    
    while np.any(children[node] != -2) and not f:
        move_mask = t_nonfatal_moves(state)

        if np.sum(move_mask) == 0:
            return state, -1, False, node

        possible_moves = children[node, move_mask]
        
        q = -stats[possible_moves, 2] 
        
        # sqrt can be slow, fastmath helps here
        sqrt_visits = np.sqrt(stats[node, 0])
        
        u = c * stats[possible_moves, 3] * (sqrt_visits / (stats[possible_moves, 0] + 1))
        
        move_idx = np.argmax(u + q)

        # Use global tuple for direction lookup
        dr, dc = DIRECTIONS_TUPLE[0], DIRECTIONS_TUPLE[1] # Dummy init to help inference if needed
        
        current_valid_count = -1
        real_move_idx = 0
        for i in range(4):
            if move_mask[i]:
                current_valid_count += 1
                if current_valid_count == move_idx:
                    real_move_idx = i
                    break
        
        move = np.array(DIRECTIONS_TUPLE[real_move_idx], dtype=np.int16)

        state, r, f = t_get_next_state_food_info(state, move)
        state = t_flip_player(state)

        node = possible_moves[move_idx]

    return state, r, f, node

@njit(types.float32(GameStateType), fastmath=True)
def playout(state):
    player = 1
    
    # 50 steps deep simulation
    for _ in range(50):
        move_mask = t_nonfatal_moves(state)
        
        # Check Loss
        if np.sum(move_mask) == 0:
            return -player

        move_idx = get_best_greedy_move(state, move_mask)

        move_vec = np.array(DIRECTIONS_TUPLE[move_idx], dtype=np.int16)
        
        state, r = t_move(state, move_vec)
        
        if r != 0:
            return player * r
            
        state = t_flip_player(state)
        player = -player

    snake0_len = state[2]
    snake1_len = state[5]
    snake0_hp = state[3]
    snake1_hp = state[6]
    
    return 0.1 * player * np.sign((snake0_len - snake1_len)) + 0.005 * player * (snake0_hp - snake1_hp)


@njit((types.int32, types.float32, types.Array(types.float32, 2, 'C', False, aligned=True), types.Array(types.int32, 1, 'C', False, aligned=True)), fastmath=True)
def backpropagate(node, result, stats, parent):
    while node > 0:
        stats[node, 0] += 1
        stats[node, 1] += result
        stats[node, 2] = stats[node, 1] / stats[node, 0]

        result = -result
        node = parent[node]
    
    stats[0, 0] += 1
    stats[0, 1] += result
    stats[0, 2] = stats[0, 1] / stats[0, 0]

@njit((types.int32, types.Array(types.float32, 1, 'C', False, aligned=True), types.Array(types.float32, 2, 'C', False, aligned=True), types.Array(types.int32, 2, 'C', False, aligned=True), types.Array(types.int32, 1, 'C', False, aligned=True), types.int32), fastmath=True)
def update_tree(node, p, stats, children, parent, n_nodes):
    children_nodes = np.arange(4) + n_nodes
    
    stats[children_nodes, 3] = p
    parent[children_nodes] = node
    children[node] = children_nodes

@njit
def get_priors(state, temp = 1.0, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3):
    valid_mask = t_nonfatal_moves(state)
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)
    h_val = np.zeros(4, dtype = np.float32)

    if np.sum(valid_mask) == 0:
        return h_val

    for i in range(4):
        if not valid_mask[i]:
            continue
        move = directions[i]
        next_state, r = t_get_next_state(state, move)
        h = flood_fill_heuristic(next_state, r, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3)
        h_val[i] = h

    if np.any(h_val[valid_mask] == np.inf):
        h_val = ((h_val == np.inf) / np.sum(h_val == np.inf)).astype(np.float32)
    elif np.all(h_val[valid_mask] == -np.inf) or np.sum(np.exp(h_val[valid_mask] - np.max(h_val[valid_mask]))) <= 1e-5:
        h_val[valid_mask] = 1 / np.sum(valid_mask)
    else:
        exp_vals = np.exp((h_val[valid_mask] - np.max(h_val[valid_mask])) / temp)
        h_val[valid_mask] = exp_vals / np.sum(exp_vals)
    
    return h_val

@njit
def mcts_search_heuristic_playout(root_state, num_sims: int, c: float, temp=1, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3):
    max_nodes = num_sims * 4 + 5

    parent = np.full(max_nodes, -1, np.int32)
    stats = np.full((max_nodes, 4), 0, np.float32)
    children = np.full((max_nodes, 4), -2, np.int32)

    n_nodes = 1

    move_mask = t_nonfatal_moves(root_state)
    if np.sum(move_mask) == 0:
        return stats[:5]

    v0, p0 = 0, get_priors(root_state, temp, k1, k2, k3, k4, k5)

    stats[0, 0] += 1
    stats[0, 1] += v0

    update_tree(0, p0, stats, children, parent, 1)

    n_nodes += 4

    for sim in range(num_sims):      
        selected_state, r, f, node = node_select(root_state, stats, children, c)

        if r != 0:
            backpropagate(node, r, stats, parent)

        elif f:
            v = playout(selected_state)
            backpropagate(node, -v, stats, parent)

        else:
            v, p = playout(selected_state), get_priors(selected_state, temp, k1, k2, k3, k4, k5)
            update_tree(node, p, stats, children, parent, n_nodes)
            n_nodes += 4
            backpropagate(node, -v, stats, parent)

    return stats[:5]

@njit(fastmath=True)
def mcts_search_playout(root_state, num_sims: int, c: float, temp=1):
    max_nodes = num_sims * 4 + 5

    parent = np.full(max_nodes, -1, np.int32)
    stats = np.full((max_nodes, 4), 0, np.float32)
    children = np.full((max_nodes, 4), -2, np.int32)

    n_nodes = 1

    move_mask = t_nonfatal_moves(root_state)
    if np.sum(move_mask) == 0:
        return stats[:5]

    priors = np.zeros(4, dtype = np.float32)
    priors[move_mask] = 1 / np.sum(move_mask)
    v0, p0 = 0, priors

    stats[0, 0] += 1
    stats[0, 1] += v0

    update_tree(0, p0, stats, children, parent, 1)

    n_nodes += 4

    for sim in range(num_sims):      
        selected_state, r, f, node = node_select(root_state, stats, children, c)

        if r != 0:
            backpropagate(node, -r, stats, parent)

        elif f:
            v = playout(selected_state)
            backpropagate(node, v, stats, parent)

        else:
            priors = np.zeros(4, dtype = np.float32)
            priors[move_mask] = 1 / np.sum(move_mask)
            v, p = playout(selected_state), priors
            update_tree(node, p, stats, children, parent, n_nodes)
            n_nodes += 4
            backpropagate(node, v, stats, parent)

    return stats[:5]

class AI:
    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def get_move(self):
        print(':(')
        return self.game_state.get_valid_moves()[0] if self.game_state.get_valid_moves() else np.array([1, 0], dtype = np.int16)
    
class HeuristicAI(AI):
    def __init__(self, game_state: GameState, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3):
        super().__init__(game_state)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5

    def get_move(self):
        state = self.game_state.to_tuple()
        valid_moves = t_get_valid_moves(state)

        best_h = -np.inf
        best_move = None
        for move in valid_moves:
            next_state, r = t_get_next_state(state, move)
            h = flood_fill_heuristic(next_state, r, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3)
            if h > best_h:
                best_h = h
                best_move = move

        if best_move is None:
            best_move = np.array([1, 0], np.int16)
        return best_move
    
    
class AlphaBetaAI(HeuristicAI):
    def __init__(self, game_state: GameState, depth=5, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3):
        super().__init__(game_state, k1, k2, k3, k4, k5)
        self.depth = depth

    def get_move(self):
        state = self.game_state.to_tuple()
        valid_moves = t_get_valid_moves(state)

        best_move = None
        best_score = -np.inf

        for move in valid_moves:
            next_state, r = t_get_next_state(state, move)
            score = negamax(next_state, r, self.depth - 1, -np.inf, -best_score, self.k1, self.k2, self.k3, self.k4, self.k5)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move is not None else np.array([1, 0], dtype = np.int16)
    
class MCTS(HeuristicAI):
    def __init__(self, game_state: GameState, num_simulations=250, c=1.4, temp = 1.0, k1=1.1, k2=0.2, k3=1.6, k4=0.5, k5=2.3):
        super().__init__(game_state, k1, k2, k3, k4, k5)
        self.num_simulations = num_simulations
        self.c = c
        self.temp = temp
    def get_move(self):
        state = self.game_state.to_tuple()
        # stats = mcts_search_heuristic_playout(state, self.num_simulations, self.c, self.temp, self.k1, self.k2, self.k3, self.k4, self.k5)
        stats = mcts_search_playout(state, self.num_simulations, self.c, self.temp)
        if np.any(stats[0] != 0):
            child_visits = stats[1:5, 0]
            pi = child_visits / np.sum(child_visits)
        else:
            return np.array([1, 0], dtype=np.int16)
        action_idx = np.argmax(pi)
        # Convert action index to direction
        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)
        direction = directions[action_idx]
        return direction