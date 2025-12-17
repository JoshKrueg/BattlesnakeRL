"""
Module: PPO Trainer with Visualization
Purpose: Train PPO agent with MCTS bootstrap and real-time Pygame visualization.
Updated: Fixes multiple Pygame init calls and ensures clean process shutdown.
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
import time
from copy import deepcopy

# Imports
from ai_neural_network import AZ_Learning_Network
from game_state import GameState
from game_state_optimized import t_move, t_flip_player, t_get_valid_moves
from ai_opponent import mcts_search_heuristic_playout
from snake_player import GameStateVisualizer # For visualization
import pygame

# --- CONFIG ---
CONFIG = {
    "lr": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "batch_size": 1024,
    "rollout_steps": 256,
    "num_envs": 32,             # For 9950X3D
    "update_epochs": 4,
    "bootstrap_games": 200,
    "decay_steps": 2_000_000,
    "init_food_reward": 0.5,
    "final_food_reward": 0.1,
    "win_reward": 1.0,
    "render_freq": 50,          # Render 1 game every N updates
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Opponent Settings
    "opponent_type": "self",    # Options: "mcts", "self", "previous"
    "mcts_sims": 50,            # Lower sims for speed during training
    "history_size": 5           # Number of past models to keep for "previous" opponent
}

# --- PPO AGENT ---
class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = AZ_Learning_Network()
        
    def get_action(self, board, deterministic=False):
        # Network inputs: Board only
        # Ensure input is tensor on correct device
        if not isinstance(board, torch.Tensor):
            board = torch.tensor(board, dtype=torch.float32).to(CONFIG["device"])
            
        value, logits = self.network(board)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        return action, log_prob, value

# --- OPPONENT AGENTS ---
class OpponentWrapper:
    """
    Wraps different opponent types (MCTS, PPO) into a common interface.
    Returns: action_index (int)
    """
    def __init__(self, mode, model_state=None):
        self.mode = mode
        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)
        
        if mode in ["self", "previous"]:
            self.net = AZ_Learning_Network().to("cpu") # Run opponent on CPU to save GPU for training
            if model_state:
                self.net.load_state_dict(model_state)
            self.net.eval()

    def get_move(self, state_tuple):
        if self.mode == "mcts":
            stats = mcts_search_heuristic_playout(
                state_tuple, CONFIG["mcts_sims"], 
                k1=0.7, k2=0.15, k3=1.2, k4=0.2, k5=1.9, 
                c=4.3, temp=0.09
            )
            if np.any(stats[0] != 0):
                child_visits = stats[1:5, 0]
                total = np.sum(child_visits) + 1e-8
                pi = child_visits / total
                return np.argmax(pi)
            return 0
            
        elif self.mode in ["self", "previous"]:
            # Neural Net Inference
            board = torch.tensor(state_tuple[0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, logits = self.net(board)
                probs = torch.softmax(logits, dim=1)
                # Sample for variety
                dist = Categorical(probs)
                action = dist.sample().item()
            return action
            
        # Fallback Random Valid
        valid = t_get_valid_moves(state_tuple)
        if len(valid) > 0:
            return 0 # Should pick valid but falling back for speed
        return 0

# --- BOOTSTRAPPING WORKER ---
def bootstrap_worker(num_games, worker_id):
    """
    Generates expert data using the heuristic MCTS logic.
    Worker 0 visualizes the first game if requested.
    """
    data_log = []
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)
    
    # Visualization setup for Worker 0
    visualize = (worker_id == 0)
    vis_game = None
    
    if visualize:
        try:
            pygame.init()
            vis_game = GameStateVisualizer()
        except Exception as e:
            print(f"Warning: Bootstrap visualization failed init: {e}")
            visualize = False
    
    for i in range(num_games):
        if visualize and i == 0:
            game = vis_game
            game.setup()
        else:
            game = GameState()
            game.setup()
            
        state = game.to_tuple()
        done = False
        
        while not done:
            if visualize and i == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        visualize = False
                        pygame.quit()
                
                if visualize:
                    game.draw()
                    pygame.display.flip()

            # --- Agent Turn (MCTS Expert) ---
            stats = mcts_search_heuristic_playout(state, 100, k1=1.5, k2=0.6, k3=2.3, k4=0.15, k5=1.9, c=2, temp=0.09)
            
            if np.any(stats[0] != 0):
                child_visits = stats[1:5, 0]
                total = np.sum(child_visits) + 1e-8
                pi = child_visits / total
                action_idx = np.argmax(pi)
            else:
                action_idx = 0 
                
            move = directions[action_idx]
            
            # Record Data
            data_log.append((state[0].copy(), action_idx))
            
            # Apply Move (Agent)
            state, res = t_move(state, move)
            if res != 0: 
                done = True
                break
                
            # --- Opponent Turn (Simulated) ---
            state = t_flip_player(state)
            
            # Use MCTS for opponent too in bootstrap for realistic games
            opp_stats = mcts_search_heuristic_playout(state, 100, k1=1.5, k2=0.6, k3=2.3, k4=0.15, k5=1.9, c=2, temp=0.09)
            
            if np.any(opp_stats[0] != 0):
                child_visits = opp_stats[1:5, 0]
                pi = child_visits / (np.sum(child_visits) + 1e-8)
                opp_idx = np.argmax(pi)
            else:
                opp_idx = 0
            
            opp_move = directions[opp_idx]
            state, opp_res = t_move(state, opp_move)
            
            state = t_flip_player(state)
            
            if opp_res != 0:
                done = True
                
            if visualize and i == 0:
                game.gameboard = state[0]
            
    if visualize:
        pygame.quit()
            
    return data_log

def run_bootstrap(agent):
    print("--- Phase 1: Bootstrapping (Imitation Learning) ---")
    
    num_workers = max(1, mp.cpu_count() - 2)
    per_worker = CONFIG["bootstrap_games"] // num_workers
    
    args = [(per_worker, i) for i in range(num_workers)]
    
    # We use spawn to ensure clean process start, reducing pygame conflicts
    # However, 'spawn' is slower to start. 'fork' (default on linux) copies memory.
    # Pygame often dislikes fork.
    
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(bootstrap_worker, args)
        
    flat_data = [item for sublist in results for item in sublist]
    print(f"Generated {len(flat_data)} expert samples.")
    
    agent.to(CONFIG["device"])
    agent.network.train()
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["lr"])
    ce_loss = nn.CrossEntropyLoss()
    
    boards = torch.tensor(np.array([x[0] for x in flat_data]), dtype=torch.float32).to(CONFIG["device"])
    targets = torch.tensor(np.array([x[1] for x in flat_data]), dtype=torch.long).to(CONFIG["device"])
    
    bs = 512
    for e in range(5):
        perm = torch.randperm(len(boards))
        ep_loss = 0
        for i in range(0, len(boards), bs):
            idx = perm[i:i+bs]
            _, logits = agent.network(boards[idx])
            loss = ce_loss(logits, targets[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        print(f"Bootstrap Epoch {e+1} Loss: {ep_loss / (len(boards)/bs):.4f}")
        
    torch.save(agent.network.state_dict(), "models/pretrained_ppo.pt")
    print("Bootstrapping Complete. Switching to PPO...")

# --- PPO WORKER (With Optional View) ---
def ppo_env_worker(pipe, start_step, worker_id, opponent_config):
    """
    Runs game environment. Worker 0 can render.
    opponent_config: tuple (mode, model_state_dict)
    """
    dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int16)
    local_step = start_step
    
    # Initialize Opponent
    opp_mode, opp_state = opponent_config
    opponent = OpponentWrapper(opp_mode, opp_state)
    
    visualize = False
    vis_game = None

    game = GameState()
    game.setup()
    state = game.to_tuple()
    
    while True:
        try:
            if pipe.poll(0.001): # Non-blocking check primarily
                cmd, data = pipe.recv()
            else:
                # If no command, wait (blocking)
                cmd, data = pipe.recv()
        except EOFError:
            break
        
        if cmd == 'reset':
            game = GameState()
            game.setup()
            state = game.to_tuple()
            if visualize and vis_game: 
                vis_game.gameboard = state[0].copy()
                vis_game.food_coords = state[1].copy()
                vis_game.snake0_coords = state[2].copy()
                vis_game.snake1_coords = state[5].copy()
            pipe.send(state[0])
            
        elif cmd == 'update_opponent':
            # Receive new model state for self-play
            new_state = data
            if opponent.mode in ["self", "previous"]:
                opponent.net.load_state_dict(new_state)
                
        elif cmd == 'enable_render':
            if worker_id == 0 and not visualize:
                try:
                    pygame.init()
                    vis_game = GameStateVisualizer()
                    vis_game.gameboard = state[0].copy()
                    visualize = True
                except:
                    visualize = False
        
        elif cmd == 'disable_render':
            if worker_id == 0 and visualize:
                visualize = False
                if vis_game: 
                    pygame.quit()
                    vis_game = None

        elif cmd == 'step':
            action_idx = data
            move = dirs[action_idx]
            local_step += 1
            
            # --- AGENT MOVE ---
            next_state, res = t_move(state, move)
            
            done = False
            reward = 0.01
            decay = min(1.0, local_step / CONFIG["decay_steps"])
            food_r = CONFIG["init_food_reward"] - decay * (CONFIG["init_food_reward"] - CONFIG["final_food_reward"])
            
            if res == -1:
                reward = -1.0
                done = True
            elif res == 1:
                reward = CONFIG["win_reward"]
                done = True
            elif next_state[3] > state[3]:
                reward += food_r
                
            # --- OPPONENT MOVE ---
            if not done:
                opp_state = t_flip_player(next_state)
                
                # Get move from configured opponent
                opp_act_idx = opponent.get_move(opp_state)
                opp_move = dirs[opp_act_idx]
                
                opp_next_state, opp_res = t_move(opp_state, opp_move)
                next_state = t_flip_player(opp_next_state)
                
                if opp_res == -1: # Opponent died
                    reward = CONFIG["win_reward"]
                    done = True
            
            state = next_state
            
            if visualize and vis_game:
                # Pump event queue to keep window alive
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT:
                        visualize = False
                        pygame.quit()
                
                if visualize:
                    vis_game.gameboard = state[0]
                    # Note: We aren't syncing full object state (coords) here for speed
                    # GameStateVisualizer.draw() usually only needs gameboard array
                    vis_game.draw()
                    pygame.display.flip()
            
            if done:
                game = GameState()
                game.setup()
                state = game.to_tuple()
                
            pipe.send((state[0], reward, done, local_step))
            
        elif cmd == 'close':
            if visualize: pygame.quit()
            break

class ViewableVectorizedEnv:
    def __init__(self, num_envs, agent_model_state):
        self.conns = []
        self.ps = []
        self.step_cnt = 0
        
        # Configure Opponent Strategy
        opp_mode = CONFIG["opponent_type"]
        opp_config = (opp_mode, agent_model_state if opp_mode != "mcts" else None)
        
        for i in range(num_envs):
            parent, child = mp.Pipe()
            p = mp.Process(target=ppo_env_worker, args=(child, 0, i, opp_config))
            p.start()
            self.conns.append(parent)
            self.ps.append(p)
            
    def update_opponent_model(self, model_state):
        for c in self.conns:
            c.send(('update_opponent', model_state))
            
    def set_render(self, enable=True):
        msg = 'enable_render' if enable else 'disable_render'
        # Only send to worker 0
        self.conns[0].send((msg, None))
            
    def reset(self):
        boards = []
        for c in self.conns:
            c.send(('reset', None))
        for c in self.conns:
            boards.append(c.recv())
        return np.array(boards)
        
    def step(self, actions):
        for i, c in enumerate(self.conns):
            c.send(('step', actions[i]))
            
        boards, rewards, dones = [], [], []
        max_s = 0
        for c in self.conns:
            b, r, d, s = c.recv()
            boards.append(b)
            rewards.append(r)
            dones.append(d)
            if s > max_s: max_s = s
        self.step_cnt = max_s
        return np.array(boards), np.array(rewards), np.array(dones)
        
    def close(self):
        for c in self.conns: c.send(('close', None))
        for p in self.ps: p.join()

# --- MAIN LOOP ---
def train_ppo_view():
    # Force spawn to ensure cleaner process isolation for Pygame
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    if not os.path.exists("models"): os.makedirs("models")
    
    agent = PPOAgent().to(CONFIG["device"])
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["lr"])
    
    # 1. Bootstrap
    if CONFIG["bootstrap_games"] > 0:
        if os.path.exists("models/pretrained_ppo.pt"):
            print("Loading pretrained model...")
            agent.network.load_state_dict(torch.load("models/pretrained_ppo.pt"))
        else:
            run_bootstrap(agent)
            
    # 2. PPO
    print(f"--- Phase 2: PPO Training (Opponent: {CONFIG['opponent_type']}) ---")
    
    # Get initial state for workers
    initial_state = agent.network.state_dict()
    # Ensure CPU for pickling
    cpu_state = {k: v.cpu() for k, v in initial_state.items()}
    
    envs = ViewableVectorizedEnv(CONFIG["num_envs"], cpu_state)
    obs = envs.reset()
    
    # History for "previous" opponent
    model_history = [cpu_state]
    
    num_updates = int(CONFIG["total_timesteps"] / (CONFIG["num_envs"] * CONFIG["rollout_steps"]))
    
    for update in range(num_updates):
        
        # Periodic Opponent Update
        if CONFIG["opponent_type"] == "self" and update % 5 == 0:
            # Update workers with latest agent
            current_state = {k: v.cpu() for k, v in agent.network.state_dict().items()}
            envs.update_opponent_model(current_state)
            
        elif CONFIG["opponent_type"] == "previous" and update % 20 == 0:
            # Pick random past model
            current_state = {k: v.cpu() for k, v in agent.network.state_dict().items()}
            model_history.append(current_state)
            if len(model_history) > CONFIG["history_size"]:
                model_history.pop(0)
            
            random_past = random.choice(model_history)
            envs.update_opponent_model(random_past)
        
        # Render Check
        rendering = ((update % CONFIG["render_freq"]) == 0)
        if rendering:
            print(f">> Rendering update {update}...")
            envs.set_render(True)
        
        # Rollout buffers
        b_obs, b_act, b_logp, b_rew, b_don, b_val = [], [], [], [], [], []
        
        for _ in range(CONFIG["rollout_steps"]):
            t_obs = torch.tensor(obs, dtype=torch.float32).to(CONFIG["device"])
            
            with torch.no_grad():
                act, logp, val = agent.get_action(t_obs)
            
            next_obs, rew, don = envs.step(act.cpu().numpy())
            
            b_obs.append(t_obs)
            b_act.append(act)
            b_logp.append(logp)
            b_val.append(val)
            b_rew.append(torch.tensor(rew).to(CONFIG["device"]))
            b_don.append(torch.tensor(don).to(CONFIG["device"]))
            
            obs = next_obs
            
        if rendering:
            envs.set_render(False)
            
        # GAE
        with torch.no_grad():
            t_next = torch.tensor(next_obs, dtype=torch.float32).to(CONFIG["device"])
            _, _, next_val = agent.get_action(t_next)
            
        returns = []
        gae = 0
        for t in reversed(range(CONFIG["rollout_steps"])):
            if t == CONFIG["rollout_steps"] - 1:
                next_non_terminal = 1.0 - torch.tensor(don, dtype=torch.float32).to(CONFIG["device"])
                next_value = next_val.flatten()
            else:
                next_non_terminal = 1.0 - b_don[t+1].float()
                next_value = b_val[t+1].flatten()
                
            delta = b_rew[t] + CONFIG["gamma"] * next_value * next_non_terminal - b_val[t].flatten()
            gae = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * next_non_terminal * gae
            returns.insert(0, gae + b_val[t].flatten())
            
        # Flatten & Train
        f_obs = torch.cat(b_obs)
        f_act = torch.cat(b_act)
        f_logp = torch.cat(b_logp)
        f_ret = torch.cat(returns)
        f_adv = f_ret - torch.cat(b_val).flatten()
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        
        agent.network.train()
        inds = np.arange(f_obs.size(0))
        for _ in range(CONFIG["update_epochs"]):
            np.random.shuffle(inds)
            for start in range(0, len(inds), CONFIG["batch_size"]):
                end = start + CONFIG["batch_size"]
                mb = inds[start:end]
                
                _, logits = agent.network(f_obs[mb])
                curr_val = agent.network(f_obs[mb])[0].view(-1)
                
                probs = torch.softmax(logits, dim=1)
                dist = Categorical(probs)
                curr_logp = dist.log_prob(f_act[mb])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(curr_logp - f_logp[mb])
                surr1 = ratio * f_adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - CONFIG["clip_eps"], 1.0 + CONFIG["clip_eps"]) * f_adv[mb]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (f_ret[mb] - curr_val).pow(2).mean()
                
                loss = actor_loss + CONFIG["vf_coef"] * value_loss - CONFIG["ent_coef"] * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if update % 5 == 0:
            mean_rew = torch.stack(b_rew).mean().item()
            print(f"Update {update} | Avg Step Reward: {mean_rew:.4f} | Loss: {loss.item():.4f}")
            
        if update % 20 == 0:
            torch.save(agent.network.state_dict(), f"models/ppo_iter_{update}.pt")
            
    envs.close()

if __name__ == "__main__":
    train_ppo_view()