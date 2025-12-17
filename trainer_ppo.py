"""
Module: PPO Trainer with MCTS Bootstrap
Purpose: Train using PPO, jump-started by imitation learning from MCTS.
Optimized: Hybrid Training (Self-Play + vs MCTS) for max CPU/GPU utilization.
Includes: Action Masking to prevent invalid moves and correct entropy calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np
import os
import random
from tqdm import tqdm

# Imports
from ai_neural_network import AZ_Learning_Network
from game_state import GameState 
# Import optimized game functions
from game_state_optimized import t_get_next_state, t_get_valid_moves, t_move, t_flip_player, t_nonfatal_moves
# Import MCTS for the opponent
from ai_opponent import mcts_search_heuristic_playout, mcts_search_playout

torch.set_float32_matmul_precision('high')

# --- HYPERPARAMETERS ---
CONFIG = {
    "num_envs": 128,             # Utilizing your 9950X3D
    "rollout_steps": 256,       # Steps per env per update
    "total_timesteps": 128*256*250,
    "lr": 2e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.002,
    "vf_coef": 0.5,
    "batch_size": 1012,
    "epochs": 5,
    "bootstrap_games": 512,     
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Reward Config
    "initial_food_reward": 0.1, 
    "final_food_reward": 0.1,   
    "decay_steps": 2_000_000,   
    "win_reward": 1.0,

    # Opponent Config
    "mcts_ratio": 0.1,          # 20% of games are vs MCTS, 80% Self-Play
    "mcts_sims_train": 64       # Low sim count for fast training opponent
}

# --- ENVIRONMENT WORKER ---
def env_worker(pipe_conn, worker_id, start_step_count, mode="self_play"):
    """
    Worker that runs the game.
    mode: 'self_play' (Agent plays both sides) or 'vs_mcts' (Agent vs Heuristic MCTS)
    """
    # 1. PRE-ALLOCATION
    DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)
    
    # 2. LOCAL VARS
    DECAY_STEPS = CONFIG["decay_steps"]
    INIT_FOOD_R = CONFIG["initial_food_reward"]
    FINAL_FOOD_R = CONFIG["final_food_reward"]
    WIN_R = CONFIG["win_reward"]
    DECAY_RANGE = INIT_FOOD_R - FINAL_FOOD_R
    MCTS_SIMS = CONFIG["mcts_sims_train"]

    # 3. SETUP
    game_ref = GameState()
    game_ref.setup()
    state_tuple = game_ref.to_tuple()
    
    local_step = start_step_count
    current_game_len = 0
    
    # Helper to get mask
    def get_mask(s):
        return t_nonfatal_moves(s)

    while True:
        cmd, data = pipe_conn.recv()
        
        if cmd == 'step':
            # == PLAYER 0 TURN (PPO AGENT) ==
            move_vec = DIRS[data] 
            local_step += 1
            current_game_len += 1
            
            # Execute PPO Move
            next_state_tuple, result = t_move(state_tuple, move_vec)
            
            # --- Reward Calculation ---
            reward = 0.0
            done = False
            
            if result == -1:
                reward = -1.0
                done = True
            elif result == 1:
                reward = WIN_R
                done = True
            
            # Food Reward
            if local_step < DECAY_STEPS:
                progress = local_step / DECAY_STEPS
                current_food_reward = INIT_FOOD_R - (progress * DECAY_RANGE)
            else:
                current_food_reward = FINAL_FOOD_R

            if not done and next_state_tuple[3] > state_tuple[3]:
                reward = current_food_reward

            # == OPPONENT TURN HANDLING ==
            if not done:
                # Flip to Player 1 Perspective
                p1_state = t_flip_player(next_state_tuple)
                
                if mode == "self_play":
                    # In self-play, we stop here and return P1 state to PPO Agent.
                    # The PPO agent will generate the move for P1 in the next step.
                    state_tuple = p1_state
                
                elif mode == "vs_mcts":
                    # In vs_mcts, the CPU calculates the P1 move immediately.
                    
                    # 1. Run MCTS for Opponent
                    # Note: We use the heuristic playout which is Numba optimized
                    stats = mcts_search_playout(p1_state, num_sims=MCTS_SIMS, c=1.4, temp=1)
                    
                    if np.any(stats[:, 0] != 0):
                        child_visits = stats[1:5, 0]
                        # Greedy selection for training opponent
                        action_idx = np.argmax(child_visits)
                    else:
                        action_idx = 0
                        
                    p1_move_vec = DIRS[action_idx]
                    
                    # 2. Execute Opponent Move
                    p1_next_state, p1_result = t_move(p1_state, p1_move_vec)
                    
                    # 3. Check Result for Opponent
                    if p1_result == -1:
                        # Opponent died -> PPO Agent Wins
                        reward = WIN_R
                        done = True
                    elif p1_result == 1:
                        # Opponent won -> PPO Agent Lost (Died)
                        reward = -1.0
                        done = True
                    else:
                        # 4. Flip back to P0 Perspective for next step
                        state_tuple = t_flip_player(p1_next_state)

            # Prepare Info
            info = {'global_step': local_step}
            if done:
                info['game_len'] = current_game_len
                current_game_len = 0
                
                # --- MCTS DEBUG LOGIC ---
                if mode == "vs_mcts":
                    info['mcts_outcome'] = 1 if reward >= (WIN_R - 0.01) else 0
                
                # Reset
                game_ref.setup() 
                state_tuple = game_ref.to_tuple()

            # Calculate Valid Move Mask for the NEXT state
            valid_mask = get_mask(state_tuple)

            pipe_conn.send((state_tuple[0], reward, done, info, valid_mask))

        elif cmd == 'reset':
            game_ref.setup()
            state_tuple = game_ref.to_tuple()
            current_game_len = 0
            
            # Calculate Valid Move Mask for initial state
            valid_mask = get_mask(state_tuple)
            
            pipe_conn.send((state_tuple[0], valid_mask))
            
        elif cmd == 'close':
            pipe_conn.close()
            break

class VectorizedSnake:
    def __init__(self, num_envs, start_step=0):
        self.num_envs = num_envs
        self.ps = []
        self.conns = []
        self.step_count = start_step
        
        # Calculate split
        num_mcts = int(num_envs * CONFIG["mcts_ratio"])
        
        print(f"Initializing Envs: {num_envs - num_mcts} Self-Play | {num_mcts} vs MCTS")
        
        for i in range(num_envs):
            # randomly assign mode based on split
            mode = "vs_mcts" if i < num_mcts else "self_play"
            
            parent, child = mp.Pipe()
            p = mp.Process(target=env_worker, args=(child, i, start_step, mode))
            p.start()
            self.ps.append(p)
            self.conns.append(parent)
            
    def reset(self):
        boards = []
        masks = []
        for conn in self.conns:
            conn.send(('reset', None))
        for conn in self.conns:
            b, m = conn.recv()
            boards.append(b)
            masks.append(m)
        return np.array(boards), np.array(masks, dtype=np.bool_)
    
    def step(self, actions):
        for i, conn in enumerate(self.conns):
            conn.send(('step', actions[i]))
            
        boards, rewards, dones, infos, masks = [], [], [], [], []
        max_step = 0 
        
        for conn in self.conns:
            b, r, d, info, m = conn.recv()
            boards.append(b)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
            masks.append(m)
            if info['global_step'] > max_step: max_step = info['global_step']
            
        self.step_count = max_step
            
        return (np.array(boards), 
                np.array(rewards, dtype=np.float32), 
                np.array(dones, dtype=np.bool_),
                infos,
                np.array(masks, dtype=np.bool_))
                
    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for p in self.ps:
            p.join()

# --- PPO AGENT ---
class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = AZ_Learning_Network()
        
    def get_action(self, board, action_mask=None, deterministic=False):
        value, logits = self.network(board)
        
        # --- ACTION MASKING ---
        if action_mask is not None:
            # Handle edge case where all moves are invalid (trapped)
            # Unmask all to allow the network to pick its preferred death
            all_false = ~torch.any(action_mask, dim=1)
            if torch.any(all_false):
                action_mask[all_false] = True
            
            # Apply mask: Set invalid moves to -1e8 (effectively 0 prob)
            logits = logits.masked_fill(~action_mask, -1e8)
            
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        return action, log_prob, value

# --- BOOTSTRAPPING ---
def bootstrap_worker(num_games):
    data_log = []
    dirs = [np.array([1, 0], dtype=np.int16), 
            np.array([0, 1], dtype=np.int16), 
            np.array([-1, 0], dtype=np.int16), 
            np.array([0, -1], dtype=np.int16)]
    
    for _ in range(num_games):
        game_state = GameState()
        game_state.setup()
        state = game_state.to_tuple()
        done = False
        trajectory = []
        
        while not done:
            stats = mcts_search_playout(state, num_sims=512, c=1.4, temp = 0.1)
            if np.any(stats[:, 0] != 0): 
                child_visits = stats[1:5, 0] 
                pi = child_visits / (np.sum(child_visits) + 1e-8)
                action_idx = np.argmax(pi)
            else:
                action_idx = 0 
            
            move_vec = dirs[action_idx]
            trajectory.append((state[0].copy(), action_idx))
            
            next_state_tuple, result = t_move(state, move_vec)
            next_state_tuple =  t_flip_player(next_state_tuple)
            
            if result != 0:
                done = True
            state = next_state_tuple

        final_result = float(result)
        current_val_sign = 1.0
        trajectory_len = len(trajectory)
        
        for i in range(trajectory_len - 1, -1, -1):
            board_snapshot, action_snapshot = trajectory[i]
            steps_from_end = (trajectory_len - 1) - i
            decay_weight = 0.5 ** (steps_from_end / 50.0)
            target_value = final_result * current_val_sign * decay_weight
            data_log.append((board_snapshot, action_snapshot, target_value))
            current_val_sign *= -1.0
            
    return data_log

def run_bootstrap(agent):
    print(f"--- Bootstrapping ({CONFIG['bootstrap_games']} games) ---")
    num_workers = max(1, mp.cpu_count() - 2)
    games_per_worker = CONFIG["bootstrap_games"] // num_workers
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(bootstrap_worker, [games_per_worker] * num_workers)
    
    expert_data = [item for sublist in results for item in sublist]
    print(f"Generated {len(expert_data)} samples.")
    
    agent.to(CONFIG["device"])
    agent.network.train()
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["lr"])
    #Cosine annealing with warmup
    warmup_epochs = 3
    total_epochs = 30
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    batch_size = 1024
    epochs = 30
    
    all_boards = np.array([x[0] for x in expert_data])
    all_targets = np.array([x[1] for x in expert_data])
    all_values = np.array([x[2] for x in expert_data])
    
    dataset_size = len(all_boards)
    indices = np.arange(dataset_size)
    
    print("Moving bootstrap data to VRAM...")
    t_boards_all = torch.tensor(all_boards, dtype=torch.float32).to(CONFIG["device"])
    t_targets_all = torch.tensor(all_targets, dtype=torch.long).to(CONFIG["device"])
    t_values_all = torch.tensor(all_values, dtype=torch.float32).to(CONFIG["device"])
    
    for e in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0
        
        for i in range(0, dataset_size, batch_size):
            idx = indices[i:i+batch_size]
            b = t_boards_all[idx]
            t = t_targets_all[idx]
            v_target = t_values_all[idx]
            
            values_pred, logits = agent.network(b)
            loss_p = policy_loss_fn(logits, t)
            loss_v = value_loss_fn(values_pred.view(-1), v_target)
            loss = loss_p + 0.5 * loss_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {e+1}: Loss {total_loss / (dataset_size/batch_size):.4f}")
    
    if not os.path.exists("models_ppo"): os.makedirs("models_ppo")
    torch.save(agent.network.state_dict(), "models_ppo/ppo_iter_90.pt")
    print("Bootstrapping Complete.")

def augment_batch(b_board, b_act, b_log_prob, b_ret, b_adv, b_mask):
    """
    Augments the batch by adding 90, 180, and 270 degree rotations.
    """
    aug_board = [b_board]
    aug_act = [b_act]
    aug_log_prob = [b_log_prob] 
    aug_ret = [b_ret]
    aug_adv = [b_adv]
    aug_mask = [b_mask]

    for k in [1, 2, 3]: 
        rotated_board = torch.rot90(b_board, k=k, dims=[-2, -1])
        rotated_act = (b_act + k) % 4
        # Rotate mask to match action rotation
        rotated_mask = torch.roll(b_mask, shifts=k, dims=1)
        
        aug_board.append(rotated_board)
        aug_act.append(rotated_act)
        aug_log_prob.append(b_log_prob)
        aug_ret.append(b_ret)
        aug_adv.append(b_adv)
        aug_mask.append(rotated_mask)

    return (
        torch.cat(aug_board, dim=0),
        torch.cat(aug_act, dim=0),
        torch.cat(aug_log_prob, dim=0),
        torch.cat(aug_ret, dim=0),
        torch.cat(aug_adv, dim=0),
        torch.cat(aug_mask, dim=0)
    )

# --- MAIN LOOP ---
def train():
    mp.set_start_method('spawn', force=True)
    
    agent = PPOAgent().to(CONFIG["device"])
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["total_timesteps"] // (CONFIG["num_envs"] * CONFIG["rollout_steps"]))
    
    if CONFIG["bootstrap_games"] > 0:
        if os.path.exists("models_ppo/pretrained_ppo.pt"):
            print("Loading existing pretrained model...")
            agent.network.load_state_dict(torch.load("models_ppo/pretrained_ppo.pt", map_location=CONFIG["device"]))
        else:
            run_bootstrap(agent)
            
    print("--- Switching to PPO Training (Self-Play + vs MCTS) ---")
    
    envs = VectorizedSnake(CONFIG["num_envs"])
    obs_board, action_mask = envs.reset()
    
    num_updates = int(CONFIG["total_timesteps"] / (CONFIG["num_envs"] * CONFIG["rollout_steps"]))
    
    raw_count = CONFIG["num_envs"] * CONFIG["rollout_steps"]
    aug_count = raw_count * 4
    print(f"\n--- GPU WORKLOAD PREVIEW ---")
    print(f"Total Process: {aug_count:,} positions per update.")
    print(f"----------------------------\n")
    
    for update in range(num_updates):
        buffer = {'b':[], 'a':[], 'lp':[], 'r':[], 'd':[], 'v':[], 'm':[]}
        episode_lengths = []
        
        mcts_wins = 0
        mcts_games = 0
        
        for _ in range(CONFIG["rollout_steps"]):
            t_board = torch.tensor(obs_board, dtype=torch.float32).to(CONFIG["device"])
            t_mask = torch.tensor(action_mask, dtype=torch.bool).to(CONFIG["device"])
            
            with torch.no_grad():
                # PASS MASK to prevent suicide
                action, log_prob, value = agent.get_action(t_board, action_mask=t_mask)
            
            next_b, reward, done, infos, next_mask = envs.step(action.cpu().numpy())
            
            for i, info in enumerate(infos):
                if done[i] and 'game_len' in info:
                    episode_lengths.append(info['game_len'])
                
                if 'mcts_outcome' in info:
                    mcts_wins += info['mcts_outcome']
                    mcts_games += 1
            
            buffer['b'].append(t_board)
            buffer['a'].append(action)
            buffer['lp'].append(log_prob)
            buffer['v'].append(value)
            buffer['r'].append(torch.tensor(reward).to(CONFIG["device"]))
            buffer['d'].append(torch.tensor(done).to(CONFIG["device"]))
            buffer['m'].append(t_mask)
            
            obs_board = next_b
            action_mask = next_mask
            
        with torch.no_grad():
            t_nb = torch.tensor(next_b, dtype=torch.float32).to(CONFIG["device"])
            _, _, next_val = agent.get_action(t_nb)
            
        num_mcts = int(CONFIG["num_envs"] * CONFIG["mcts_ratio"])
        gae_invert = torch.ones(CONFIG["num_envs"], device=CONFIG["device"])
        if num_mcts < CONFIG["num_envs"]:
             gae_invert[num_mcts:] = -1.0

        returns, advantages = [], []
        gae = 0
        
        for t in reversed(range(CONFIG["rollout_steps"])):
            if t == CONFIG["rollout_steps"] - 1:
                nextnonterminal = 1.0 - torch.tensor(done, dtype=torch.float32).to(CONFIG["device"])
                nextvalues = next_val.flatten()
            else:
                nextnonterminal = 1.0 - buffer['d'][t].float()
                nextvalues = buffer['v'][t+1].flatten()
            
            # CRITICAL FIX:
            # For Self-Play, V(s') is from opponent perspective. We must negate it 
            # to represent the value for the current player.
            # We also negate the incoming 'gae' from the future step because 
            # A(t+1) is the advantage for the opponent.
            
            # Apply inversion to next_values
            delta = buffer['r'][t] + CONFIG["gamma"] * (nextvalues * gae_invert) * nextnonterminal - buffer['v'][t].flatten()
            
            # Apply inversion to the recursive GAE term
            gae = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * nextnonterminal * (gae * gae_invert)
            
            returns.insert(0, gae + buffer['v'][t].flatten())
            advantages.insert(0, gae)
            
        b_board = torch.cat(buffer['b'])
        b_act = torch.cat(buffer['a'])
        b_log_prob = torch.cat(buffer['lp'])
        b_ret = torch.stack(returns).view(-1)
        b_adv = torch.stack(advantages).view(-1)
        b_mask = torch.cat(buffer['m'])

        # Augment batch now includes masks
        b_board, b_act, b_log_prob, b_ret, b_adv, b_mask = augment_batch(
            b_board, b_act, b_log_prob, b_ret, b_adv, b_mask
        )

        b_inds = np.arange(b_board.size(0))
        
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_iters = 0
        
        for _ in range(CONFIG["epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), CONFIG["batch_size"]):
                end = start + CONFIG["batch_size"]
                idx = b_inds[start:end]
                
                _, logits = agent.network(b_board[idx])
                values = agent.network(b_board[idx])[0].view(-1)
                
                # --- APPLY MASK TO LOGITS BEFORE LOSS CALC ---
                batch_mask = b_mask[idx]
                all_false = ~torch.any(batch_mask, dim=1)
                if torch.any(all_false):
                    batch_mask[all_false] = True
                    
                logits = logits.masked_fill(~batch_mask, -1e8)
                
                probs = torch.softmax(logits, dim=1)
                dist = Categorical(probs)
                new_log_prob = dist.log_prob(b_act[idx])

                # Masked Entropy: This ensures the agent is only exploring VALID moves
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - b_log_prob[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - CONFIG["clip_range"], 1.0 + CONFIG["clip_range"]) * b_adv[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (b_ret[idx] - values).pow(2).mean()
                
                loss = actor_loss + CONFIG["vf_coef"] * value_loss - CONFIG["ent_coef"] * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_iters += 1
                
        scheduler.step()
        if update % 10 == 0:
            avg_rew = torch.stack(buffer['r']).mean().item()
            avg_len = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0.0
            avg_act_loss = total_actor_loss / update_iters
            avg_val_loss = total_value_loss / update_iters
            avg_ent = total_entropy / update_iters
            
            mcts_rate = (mcts_wins / mcts_games) if mcts_games > 0 else 0.0
            
            print(f"Update {update} | Avg Reward: {avg_rew:.4f} | Avg Len: {avg_len:.1f} | "
                  f"Act Loss: {avg_act_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Ent: {avg_ent:.4f} | MCTS Win%: {mcts_rate*100:.1f}%")
            
            torch.save(agent.network.state_dict(), f"models_ppo/ppo_iter_{update}.pt")
            
    envs.close()

if __name__ == "__main__":
    if not os.path.exists("models_ppo"): os.makedirs("models_ppo")
    train()