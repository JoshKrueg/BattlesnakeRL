"""
Module: Neural Network AI Models
Purpose: Deep learning models for snake AI including value and policy networks
Optimized: PyTorch-based convolutional neural networks with MCTS integration
"""

import torch
from torch import nn
from game_state import GameState
from game_state_optimized import *
import numpy as np
from ai_opponent import AI
from ai_opponent import flood_fill
from numba import njit

DIRECTIONS_TUPLE = ((1, 0), (0, 1), (-1, 0), (0, -1))

class CNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        height = 21,
        width = 21,
    ) -> None:
        
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.block(input)
        return torch.nn.ReLU()(x)


class AZ_Learning_Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1)
        self.backbone = torch.nn.Sequential(*(
            [CNBlock(64,64) for i in range(12)]
        ))
        self.scalar_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )

        # 4. Policy Head (Actor)
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1) # Reduce to 2 channels
        self.policy_fc = nn.Sequential(
            nn.Linear(2 * 21 * 21 + 32, 4),    # +32 for scalars
            nn.Softmax(dim = 1),
        )
        # 5. Value Head (Critic)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # Reduce to 1 channel
        self.value_fc = nn.Sequential(
            nn.Linear(1 * 21 * 21 + 32, 64),               # +32 for scalars
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Output -1 to 1
        )

    def get_ff_nums(self, states):
        out_tensor = torch.zeros((len(states), 3), dtype = torch.float32, device = "cuda") 
        my_health = torch.amax(states[:, 3], dim=(1,2)) / 70
        enemy_health = torch.amax(states[:, 5], dim=(1,2)) / 70
        
        my_length = torch.amax(states[:, 2], dim=(1,2))
        enemy_length = torch.amax(states[:, 4], dim=(1,2))
        len_diff = (my_length - enemy_length) / 5 # Scale factor (approx board width * 2)
        len_diff = torch.tanh(len_diff) # Squash to [-1, 1]

        out_tensor[:, 0] = my_health
        out_tensor[:, 1] = enemy_health
        out_tensor[:, 2] = len_diff

        return out_tensor

    def norm_board(self, board):
        # Normalize the game board for better learning
        norm_board = board.clone()
        norm_board[:, 3] = norm_board[:, 3] / 70.0  # Normalize snake 0 head
        norm_board[:, 5] = norm_board[:, 5] / 70.0  # Normalize snake 1 head

        my_length = torch.amax(norm_board[:, 2], dim=(1,2))
        enemy_length = torch.amax(norm_board[:, 4], dim=(1,2))
        longest_length = torch.maximum(my_length, enemy_length)
        longest_length = torch.clamp(longest_length, min=1.0)

        scale_factor = longest_length.view(-1, 1, 1)
        norm_board[:, 2] = norm_board[:, 2] / scale_factor
        norm_board[:, 4] = norm_board[:, 4] / scale_factor
        return norm_board

    def forward(self, x):
        x_n = self.norm_board(x)
        x1 = self.conv1(x_n)
        x1 = nn.functional.relu(x1)
        x2 = self.backbone(x1)

        scalars = self.get_ff_nums(x)
        scalars = self.scalar_head(scalars)

        # Policy Head
        p = nn.functional.relu(self.policy_conv(x2))
        p = nn.Flatten()(p)      # Flatten
        p = torch.cat([p, scalars], dim=1)   # Inject scalars
        policy_probs = self.policy_fc(p)

        # Value Head
        v = nn.functional.relu(self.value_conv(x2))
        v = nn.Flatten()(v)      # Flatten
        v = torch.cat([v, scalars], dim=1)   # Inject scalars
        value = self.value_fc(v)
        return value, policy_probs


njit(types.Tuple((GameStateType, types.int16, types.boolean, types.int32))(GameStateType, types.Array(types.float32, 2, 'C'), types.Array(types.int32, 2, 'C'), types.float32), fastmath=True)
def node_select(root_state, stats, children, c):
    node = 0
    state = root_state
    r = 0
    f = False
    
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


def mcts_search(root_state, num_sims: int, c: float, model, prob_temp):
    max_nodes = num_sims * 4 + 1

    parent = np.full(max_nodes, -1, np.int32)
    stats = np.full((max_nodes, 4), 0, np.float32)
    #N, W, Q, P
    # children pointers: each node up to 4 moves
    children = np.full((max_nodes, 4), -2, np.int32)

    # root initialization
    root = 0
    n_nodes = 1
    for sim in range(num_sims):
        selected_state, r, f, node = node_select(root_state, stats, children, c)        
        
        if r != 0:
            backpropagate(node, -r, stats, parent)
        
        elif f:
            with torch.no_grad():
                v0, p0 = model(torch.tensor(selected_state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                v, p = v0.item(), p0.cpu().numpy()[0]
            backpropagate(node, v, stats, parent)

        else:
            with torch.no_grad():
                v0, p0 = model(torch.tensor(selected_state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                v, p = v0.item(), p0.cpu().numpy()[0]
            update_tree(node, p, stats, children, parent, n_nodes)
            p = (p ** prob_temp) / np.sum(p ** prob_temp)
            n_nodes += 4
            backpropagate(node, v, stats, parent)
        
    return stats[:5]

def mcts_search_init(root_state, num_init_steps, c: float, model, prob_temp, parent, stats, children):
    # root initialization
    root = 0
    n_nodes = 1
    for sim in range(num_init_steps):
        selected_state, r, f, node = node_select(root_state, stats, children, c)        
        
        if r != 0:
            backpropagate(node, -r, stats, parent)
        
        elif f:
            with torch.no_grad():
                v0, p0 = model(torch.tensor(selected_state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                v, p = v0.item(), p0.cpu().numpy()[0]
            backpropagate(node, v, stats, parent)

        else:
            with torch.no_grad():
                v0, p0 = model(torch.tensor(selected_state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                v, p = v0.item(), p0.cpu().numpy()[0]
            p = (p ** prob_temp) / np.sum(p ** prob_temp)
            update_tree(node, p, stats, children, parent, n_nodes)
            n_nodes += 4
            backpropagate(node, v, stats, parent)

    return n_nodes

@njit((types.int32, types.Array(types.float32, 2, 'C', False, aligned=True), types.Array(types.int32, 1, 'C', False, aligned=True)), fastmath=True)
def apply_virtual_loss(node, stats, parent):
    """
    Applies virtual loss from the leaf UP to the root.
    Increases N and W to discourage other threads from picking this path.
    """
    # VIRTUAL_LOSS constant hoisted or passed
    V_LOSS = 1.0 # Stronger virtual loss is usually better for batching
    
    curr = node
    while curr != -1:
        stats[curr, 0] += 1        # N += 1
        stats[curr, 1] += V_LOSS   # W += 1 (Assuming W is opponent win, this discourages us)
        stats[curr, 2] = stats[curr, 1] / stats[curr, 0]
        curr = parent[curr]

@njit((types.int32, types.float32, types.Array(types.float32, 2, 'C', False, aligned=True), types.Array(types.int32, 1, 'C', False, aligned=True)), fastmath=True)
def revert_and_backpropagate(node, value, stats, parent):
    """
    Combines reverting virtual loss and applying real value in ONE pass.
    Net Change to W = Real_Value - Virtual_Loss
    Net Change to N = 0 (Because we keep the visit, just swapping virtual for real)
    """
    V_LOSS = 1.0
    
    curr = node
    while curr != -1:
        # We don't change N (stats[curr, 0]) because the virtual visit 
        # is now becoming a real visit.
        
        # Remove virtual loss, Add real value
        stats[curr, 1] = stats[curr, 1] - V_LOSS + value
        
        # Recalculate Q
        stats[curr, 2] = stats[curr, 1] / stats[curr, 0]
        
        # Flip value for parent perspective (Minimax)
        value = -value
        # Virtual loss is always positive "penalty" in stats, 
        # but we must be careful with signs. 
        # If we added V_LOSS to child, we subtracted it from parent perspective?
        # Actually, standard MCTS with stats array usually just stores 'sum of rewards'.
        # For simplicity in your code structure:
        # Your 'backpropagate' flips the sign of 'result'.
        # So we should probably revert the specific V_LOSS applied to THIS node.
        
        curr = parent[curr]

# --- 2. Update the Search Function ---

def mcts_search_p(root_state, num_sims: int, c: float, model, prob_temp):
    max_nodes = num_sims * 4 + 5

    parent = np.full(max_nodes, -1, np.int32)
    stats = np.full((max_nodes, 4), 0, np.float32)
    children = np.full((max_nodes, 4), -2, np.int32)

    # Batch buffers
    batch_size = 8
    batch_nodes = np.zeros(batch_size, dtype=np.int32)
    batch_states = torch.zeros((batch_size, 6, 21, 21), dtype=torch.float32, device="cuda")
    
    # Track which indices in the batch need processing
    active_mask = np.zeros(batch_size, dtype=np.bool_) 
    
    # Initialize Root
    n_nodes = mcts_search_init(root_state, 16, c, model, prob_temp, parent, stats, children)

    # Main Loop
    # We step by batch_size
    for _ in range((num_sims - 16) // batch_size):
        
        # 1. Selection Phase (Fill Batch)
        for i in range(batch_size):
            selected_state, r, f, node = node_select(root_state, stats, children, c)
            
            # Apply VL immediately so next selection in this loop sees it
            apply_virtual_loss(node, stats, parent)
            
            batch_nodes[i] = node
            
            if r != 0:
                revert_and_backpropagate(node, -r, stats, parent)
                active_mask[i] = False # Don't run NN inference
            
            elif f:
                # Leaf node, needs expansion
                batch_states[i] = torch.tensor(selected_state[0], dtype=torch.float32, device='cuda')
                active_mask[i] = True
            
            else:
                batch_states[i] = torch.tensor(selected_state[0], dtype=torch.float32, device='cuda')
                active_mask[i] = True

        # 2. Inference Phase (Batched)
        if np.any(active_mask):
            with torch.no_grad():
                # Only pass active states to model to avoid noise (optional, but cleaner)
                # For fixed batch size, passing all is fine, just ignore inactive outputs.
                values, policies = model(batch_states)
                values = values.cpu().numpy()
                policies = policies.cpu().numpy()

        # 3. Backpropagation Phase
        for i in range(batch_size):
            if not active_mask[i]:
                continue
                
            node = batch_nodes[i]
            v = values[i, 0]
            p = policies[i]
            
            if children[node, 0] == -2:
                # Temperature processing
                p_temp = (p ** prob_temp) / np.sum(p ** prob_temp)
                update_tree(node, p_temp, stats, children, parent, n_nodes)
                n_nodes += 4
            
            # Revert VL and Backpropagate real value
            # Note: We backpropagate -v because value is from perspective of player at 'node'
            # and stats store value for the player at 'node'.
            revert_and_backpropagate(node, v, stats, parent)

    return stats[:5]

class NNAI(AI):
    def __init__(self, game_state: GameState, model_path: str, num_simulations: int = 800, c_puct: float = 1.4, prob_temp = 1):
        """
        Initialize the Neural Network AI.
        
        Args:
            game_state: Current game state
            model_path: Path to the pretrained model checkpoint
            num_simulations: Number of MCTS simulations to run per move
            c_puct: Exploration constant for PUCT algorithm
        """
        super().__init__(game_state)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.prob_temp = prob_temp
        # Initialize the neural network
        self.model = AZ_Learning_Network().cuda()
        
        # Load the pretrained weights
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()  # Set to evaluation mode
        

    def get_move(self):
        """
        Get the best move using MCTS with neural network guidance.
        
        Returns:
            The best move direction as a numpy array [x, y]
        """
        # Convert game state to tuple representation
        state = self.game_state.to_tuple()
        
        # # Run MCTS with neural network
        stats = mcts_search_p(state, self.num_simulations, self.c_puct, self.model, self.prob_temp)
        
        # Check if there are any valid moves
        if np.any(stats[0] != 0):
            # Get visit counts for children nodes
            child_visits = stats[1:5, 0]
        
            # Select action with highest visit count
            action_idx = np.argmax(child_visits)
            
            # Convert to direction
            best_move = DIRECTIONS_TUPLE[action_idx]
        else:
            # Fallback to default move if no valid moves
            valid_moves = self.game_state.get_valid_moves()
            best_move = valid_moves[0] if valid_moves else np.array([1, 0], dtype=np.int16)

        # valid_mask = t_nonfatal_moves(state)

        # with torch.no_grad():
        #     value, policy = self.model(torch.tensor(state[0], dtype = torch.float32, device = "cuda").unsqueeze(0),
        #                                     torch.tensor([state[4], state[7]], dtype = torch.float32, device = "cuda").unsqueeze(0))
        #             # Get action probabilities from visit counts at root
        # policy = policy.cpu().numpy()
        # action_idx = np.argmax(policy[0, valid_mask])
        # print(f"Value: {value.item()}")
        # # Convert action index to direction
        # directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype = np.int16)
        # best_move = directions[valid_mask][action_idx]
        
        return best_move

class NNAI_greedy(AI):
    def __init__(self, game_state: GameState, model_path: str):
        """
        Initialize the Neural Network AI.
        
        Args:
            game_state: Current game state
            model_path: Path to the pretrained model checkpoint
            num_simulations: Number of MCTS simulations to run per move
            c_puct: Exploration constant for PUCT algorithm
        """
        super().__init__(game_state)
        # Initialize the neural network
        self.model = AZ_Learning_Network()
        self.model.cuda()
        
        # Load the pretrained weights
        checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(checkpoint)
        self.model.eval()  # Set to evaluation mode
        
        # Define move directions
        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)

    def get_move(self):
        """
        Get the best move using MCTS with neural network guidance.
        
        Returns:
            The best move direction as a numpy array [x, y]
        """
        # Convert game state to tuple representation
        state = self.game_state.to_tuple()

        valid_mask = t_nonfatal_moves(state)

        if np.sum(valid_mask) == 0:
            return np.array([1, 0], dtype = np.int16)

        with torch.no_grad():
            value, policy = self.model(torch.tensor(state[0], dtype = torch.float32, device = "cuda").unsqueeze(0))
                                          # Get action probabilities from visit counts at root

        policy = policy.cpu().numpy()
        policy[0, ~valid_mask] = 0
        action_idx = np.argmax(policy[0])
        # Convert action index to direction
        best_move = DIRECTIONS_TUPLE[action_idx]
        
        return best_move

