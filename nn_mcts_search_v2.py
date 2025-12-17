import numpy as np
import torch
from ai_neural_network import apply_virtual_loss, update_tree, AZ_Learning_Network
from ai_opponent import AI
from numba import njit, types
from game_state_optimized import *
from game_state import GameState


DIRECTIONS_TUPLE = ((1, 0), (0, 1), (-1, 0), (0, -1))

# --- 1. Global Node Selection (with Min-Max Norm) ---
@njit(types.Tuple((GameStateType, types.int16, types.boolean, types.int32))(
    GameStateType, 
    types.Array(types.float32, 2, 'C'), 
    types.Array(types.int32, 2, 'C'), 
    types.float32, 
    types.Array(types.float32, 1, 'C') # Added min_max_stats
), fastmath=True)
def node_select(root_state, stats, children, c, min_max_stats):
    node = 0
    state = root_state
    r = 0
    f = False
    
    # Extract Global Min/Max
    g_min = min_max_stats[0]
    g_max = min_max_stats[1]
    
    while np.any(children[node] != -2) and not f:
        move_mask = t_nonfatal_moves(state)

        if np.sum(move_mask) == 0:
            return state, -1, False, node

        possible_moves = children[node, move_mask]
        
        # Get value for the player at the child node (v)
        # We want to choose the child that minimizes 'v' (Parent Perspective)
        # Standard MinMax Norm: q_norm = (g_max - v) / (g_max - g_min)
        # This maps v=g_max (good for child) to 0.0 (bad for parent)
        # This maps v=g_min (bad for child) to 1.0 (good for parent)
        
        v = stats[possible_moves, 2]
        
        # Handle normalization
        if g_max > g_min:
            q_norm = (g_max - v) / (g_max - g_min)
        else:
            # If range is 0, we treat Q as neutral (0.5) to rely purely on U (Priors)
            q_norm = np.full(v.shape, 0.5, dtype=np.float32)

        sqrt_visits = np.sqrt(stats[node, 0])
        
        # u calculation now adds to normalized q
        u = c * stats[possible_moves, 3] * (sqrt_visits / (stats[possible_moves, 0] + 1))
        
        move_idx = np.argmax(u + q_norm)

        # Direction Mapping (Existing Logic)
        dr, dc = DIRECTIONS_TUPLE[0], DIRECTIONS_TUPLE[1] 
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


# --- 2. Global Backpropagation ---
@njit((types.int32, types.float32, types.Array(types.float32, 2, 'C', False, aligned=True), 
       types.Array(types.int32, 1, 'C', False, aligned=True),
       types.Array(types.float32, 1, 'C', False, aligned=True)), fastmath=True)
def backpropagate(node, result, stats, parent, min_max_stats):
    while node > 0:
        stats[node, 0] += 1
        stats[node, 1] += result
        
        # Calculate mean Q
        q = stats[node, 1] / stats[node, 0]
        stats[node, 2] = q
        
        # Update Global Min-Max
        if q < min_max_stats[0]:
            min_max_stats[0] = q
        if q > min_max_stats[1]:
            min_max_stats[1] = q

        result = -result
        node = parent[node]
    
    # Root Update
    stats[0, 0] += 1
    stats[0, 1] += result
    q_root = stats[0, 1] / stats[0, 0]
    stats[0, 2] = q_root
    
    # Update Global Min-Max for root too
    if q_root < min_max_stats[0]:
        min_max_stats[0] = q_root
    if q_root > min_max_stats[1]:
        min_max_stats[1] = q_root


# --- 3. Revert Virtual Loss + Global Update ---
@njit((types.int32, types.float32, types.Array(types.float32, 2, 'C', False, aligned=True), 
       types.Array(types.int32, 1, 'C', False, aligned=True),
       types.Array(types.float32, 1, 'C', False, aligned=True)), fastmath=True)
def revert_and_backpropagate(node, value, stats, parent, min_max_stats):
    V_LOSS = 1.0
    curr = node
    
    # Note: Value passed in is for the player at 'node'
    # We iterate UP. 
    
    current_val_est = value
    
    while curr != -1:
        # Revert VL, Add Real Value
        stats[curr, 1] = stats[curr, 1] - V_LOSS + current_val_est
        
        # Recalc Q
        q = stats[curr, 1] / stats[curr, 0]
        stats[curr, 2] = q

        # Update Global Min-Max
        if q < min_max_stats[0]:
            min_max_stats[0] = q
        if q > min_max_stats[1]:
            min_max_stats[1] = q
            
        # Prepare for parent (opponent)
        current_val_est = -current_val_est
        curr = parent[curr]

def mcts_search_init(root_state, num_init_steps, c, model, prob_temp, parent, stats, children, min_max_stats):
    # root initialization
    root = 0
    n_nodes = 1
    for sim in range(num_init_steps):
        # PASS min_max_stats
        selected_state, r, f, node = node_select(root_state, stats, children, c, min_max_stats)        
        
        if r != 0:
            backpropagate(node, r, stats, parent, min_max_stats)
        
        elif f:
            with torch.no_grad():
                v0, p0 = model(torch.tensor(selected_state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                v, p = v0.item(), p0.cpu().numpy()[0]
            backpropagate(node, v, stats, parent, min_max_stats)

        else:
            with torch.no_grad():
                v0, p0 = model(torch.tensor(selected_state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                v, p = v0.item(), p0.cpu().numpy()[0]
            
            p = (p ** prob_temp) / np.sum(p ** prob_temp)
            update_tree(node, p, stats, children, parent, n_nodes)
            n_nodes += 4
            backpropagate(node, v, stats, parent, min_max_stats)

    return n_nodes


def mcts_search_p(root_state, num_sims: int, c: float, model, prob_temp):
    max_nodes = num_sims * 4 + 5

    parent = np.full(max_nodes, -1, np.int32)
    stats = np.full((max_nodes, 4), 0, np.float32)
    children = np.full((max_nodes, 4), -2, np.int32)

    # --- NEW: Initialize Global Min-Max Stats ---
    # Index 0: Min, Index 1: Max
    # Initialized to inverted infinity so first update sets them
    min_max_stats = np.array([float('inf'), float('-inf')], dtype=np.float32)

    batch_size = 8
    batch_nodes = np.zeros(batch_size, dtype=np.int32)
    batch_states = torch.zeros((batch_size, 6, 21, 21), dtype=torch.float32, device="cuda")
    active_mask = np.zeros(batch_size, dtype=np.bool_) 
    
    # Init with min_max_stats
    n_nodes = mcts_search_init(root_state, min(64, num_sims), c, model, prob_temp, parent, stats, children, min_max_stats)

    for _ in range(min(num_sims- 64, 0) // batch_size):
        
        # 1. Selection
        for i in range(batch_size):
            # PASS min_max_stats
            selected_state, r, f, node = node_select(root_state, stats, children, c, min_max_stats)
            
            apply_virtual_loss(node, stats, parent)
            batch_nodes[i] = node
            
            if r != 0:
                # PASS min_max_stats
                revert_and_backpropagate(node, r, stats, parent, min_max_stats)
                active_mask[i] = False 
            elif f:
                batch_states[i] = torch.tensor(selected_state[0], dtype=torch.float32, device='cuda')
                active_mask[i] = True
            else:
                batch_states[i] = torch.tensor(selected_state[0], dtype=torch.float32, device='cuda')
                active_mask[i] = True

        # 2. Inference
        if np.any(active_mask):
            with torch.no_grad():
                values, policies = model(batch_states)
                values = values.cpu().numpy()
                policies = policies.cpu().numpy()

        # 3. Backpropagation
        for i in range(batch_size):
            if not active_mask[i]:
                continue
                
            node = batch_nodes[i]
            v = values[i, 0]
            p = policies[i]
            
            if children[node, 0] == -2:
                p_temp = (p ** prob_temp) / np.sum(p ** prob_temp)
                update_tree(node, p_temp, stats, children, parent, n_nodes)
                n_nodes += 4
            
            # PASS min_max_stats
            revert_and_backpropagate(node, v, stats, parent, min_max_stats)

    return stats[:5]

class NNAIv2(AI):
    # CHANGED: c_puct reduced to 1.1 (since Q is now 0-1 normalized)
    # CHANGED: prob_temp increased to 1.0 (trust the strong network)
    def __init__(self, game_state: GameState, model_path: str, num_simulations: int = 64, c_puct: float = 1.1, prob_temp = 1.0):
        super().__init__(game_state)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.prob_temp = prob_temp
        
        self.model = AZ_Learning_Network().cuda()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
    # get_move remains the same, as it calls the updated mcts_search_p
    def get_move(self):
        state = self.game_state.to_tuple()
        stats = mcts_search_p(state, self.num_simulations, self.c_puct, self.model, self.prob_temp)
        
        if np.any(stats[0] != 0):
            child_visits = stats[1:5, 0]
            action_idx = np.argmax(child_visits)
            best_move = DIRECTIONS_TUPLE[action_idx]
        else:
            valid_moves = self.game_state.get_valid_moves()
            best_move = valid_moves[0] if valid_moves else np.array([1, 0], dtype=np.int16)
        
        return best_move