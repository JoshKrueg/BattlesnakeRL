"""
Module: Tournament Manager
Purpose: Multi-agent tournament evaluation with parallel game execution
Optimized: Concurrent process pool for rapid tournament execution
"""

import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os
import time
from typing import List, Dict, Tuple, Any, Type
from game_state import GameState
from ai_opponent import AlphaBetaAI, MCTS, HeuristicAI, AI
from ai_neural_network import NNAI_greedy, NNAI
from nn_mcts_search_v2 import NNAIv2
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

def play_game_worker(player1_class, player1_params, player2_class, player2_params, max_moves=1000):
    """
    Worker function for parallel game playing.
    
    Args:
        player1_class: Class of player 1
        player1_params: Parameters for player 1
        player2_class: Class of player 2
        player2_params: Parameters for player 2
        max_moves: Maximum number of moves before declaring a draw
    
    Returns:
        tuple: (result, moves_count, player1_name, player2_name)
        where result is 1 if player1 wins, -1 if player2 wins, 0 for draw
    """
    # Create a new game state
    game_state = GameState()
    game_state.setup()
    
    # Initialize players
    player1 = player1_class(game_state, **player1_params)
    player2 = player2_class(game_state, **player2_params)
    
    # Get player names for reporting
    player1_name = f"{player1_class.__name__}_{hash_params(player1_params)}"
    player2_name = f"{player2_class.__name__}_{hash_params(player2_params)}"
    
    cur_player = 1
    move_count = 0
    
    while move_count < max_moves:  # Prevent infinite games
        if cur_player == 1:         
            move = player1.get_move()
        else:
            move = player2.get_move()
        
        result = game_state.move(move)

        if result != 0:
            return result * cur_player, move_count, player1_name, player2_name
        
        move_count += 1
        cur_player *= -1
        game_state.flip_player()

    # Draw if max moves reached
    return 0, move_count, player1_name, player2_name

def hash_params(params: Dict) -> str:
    """Create a short hash from parameter values to use in player names"""
    if "model_path" not in params:
        param_str = f"MCTS {params['num_simulations']}"
    else:
        if params['model_path'] == "models_ppo/ppo_iter_240.pt":
            model_name = "PPO Trained"
        else:
            model_name = "Bootstrapped Only"

        if "num_simulations" not in params:
            param_str = f"Greedy {model_name}"
        else:
            param_str = f"MCTS {params['num_simulations']} {model_name}"
    return param_str

class Tournament:
    def __init__(self, 
                 players: List[Tuple[Type[AI], Dict[str, Any]]],
                 games_per_matchup: int = 4,
                 max_workers: int = 8,
                 max_moves: int = 1000,
                 results_dir: str = "tournament_results"):
        """
        Initialize the tournament.
        
        Args:
            players: List of (player_class, params_dict) tuples
            games_per_matchup: Number of games to play per matchup
            max_workers: Maximum number of parallel processes
            max_moves: Maximum moves per game before declaring a draw
            results_dir: Directory to save results
        """
        self.players = players
        self.games_per_matchup = games_per_matchup
        self.max_workers = max_workers
        self.max_moves = max_moves
        self.results_dir = results_dir
        
        # Create a unique identifier for each player based on class and parameters
        self.player_ids = [f"{player_class.__name__}_{hash_params(params)}" 
                          for player_class, params in self.players]
                          
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'matchups': [],
            'outcomes': [],
            'move_counts': [],
            'player1': [],
            'player2': []
        }
        
        # Initialize win/loss/draw statistics
        self.stats = pd.DataFrame(0, 
                                 index=self.player_ids, 
                                 columns=['wins', 'losses', 'draws', 'points', 'games_played'])
                                 
    def run_tournament(self, save_results: bool = True):
        """
        Run the tournament with all players playing against each other.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            DataFrame with tournament results
        """
        print(f"Starting tournament with {len(self.players)} players")
        print(f"Each matchup will play {self.games_per_matchup} games")
        
        # Generate all matchups (each player against each other player)
        tasks = []
        for i, (player1_class, player1_params) in enumerate(self.players):
            for j, (player2_class, player2_params) in enumerate(self.players):
                # Skip playing against yourself
                if i == j:
                    continue
                    
                # Multiple games per matchup
                for _ in range(self.games_per_matchup):
                    tasks.append((player1_class, player1_params, player2_class, player2_params))
        
        random.shuffle(tasks)  # Randomize task order
        
        # Run games in parallel
        total_games = len(tasks)
        completed_games = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Start all games
            future_to_matchup = {
                executor.submit(play_game_worker, *task, self.max_moves): task 
                for task in tasks
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_matchup), total=total_games, desc="Games"):
                player1_class, player1_params, player2_class, player2_params = future_to_matchup[future]
                try:
                    result, move_count, player1_id, player2_id = future.result()
                    completed_games += 1
                    
                    # Store result details
                    self.results['matchups'].append(f"{player1_id} vs {player2_id}")
                    self.results['outcomes'].append(result)
                    self.results['move_counts'].append(move_count)
                    self.results['player1'].append(player1_id)
                    self.results['player2'].append(player2_id)
                    
                    # Update statistics
                    if result == 1:  # Player 1 wins
                        self.stats.loc[player1_id, 'wins'] += 1
                        self.stats.loc[player1_id, 'points'] += 3
                        self.stats.loc[player2_id, 'losses'] += 1
                    elif result == -1:  # Player 2 wins
                        self.stats.loc[player2_id, 'wins'] += 1
                        self.stats.loc[player2_id, 'points'] += 3
                        self.stats.loc[player1_id, 'losses'] += 1
                    else:  # Draw
                        self.stats.loc[player1_id, 'draws'] += 1
                        self.stats.loc[player2_id, 'draws'] += 1
                        self.stats.loc[player1_id, 'points'] += 1
                        self.stats.loc[player2_id, 'points'] += 1
                        
                    self.stats.loc[player1_id, 'games_played'] += 1
                    self.stats.loc[player2_id, 'games_played'] += 1
                    
                except Exception as e:
                    print(f"Error in game: {str(e)}")
        
        # Calculate duration
        duration = time.time() - start_time
        print(f"Tournament completed: {completed_games} games in {duration:.2f} seconds")
        
        # Sort by points (descending)
        self.stats = self.stats.sort_values('points', ascending=False)
        
        # Calculate win rate
        self.stats['win_rate'] = self.stats['wins'] / self.stats['games_played']
        
        # Display results
        print("\nTournament Results:")
        print(self.stats)
        
        # Create matchup matrix
        self.create_matchup_matrix()
        
        # Save results if requested
        if save_results:
            self.save_results()
            
        return self.stats
    
    def create_matchup_matrix(self):
        """Create a matrix showing head-to-head results between players"""
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(self.results)
        
        # Initialize matrix with zeros
        self.matchup_matrix = pd.DataFrame(0, 
                                          index=self.player_ids, 
                                          columns=self.player_ids)
                                          
        # Fill in matchup results
        for _, row in results_df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            outcome = row['outcomes']
            
            if outcome == 1:  # Player 1 wins
                self.matchup_matrix.loc[player1, player2] += 1
            elif outcome == -1:  # Player 2 wins
                self.matchup_matrix.loc[player2, player1] += 1
        
        return self.matchup_matrix
    
    def save_results(self):
        """Save tournament results to disk"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.results_dir}/tournament_details_{timestamp}.csv", index=False)
        
        # Save summary statistics
        self.stats.to_csv(f"{self.results_dir}/tournament_stats_{timestamp}.csv")
        
        # Save matchup matrix
        self.matchup_matrix.to_csv(f"{self.results_dir}/matchup_matrix_{timestamp}.csv")
        
        # Create and save visualizations
        self.create_visualizations(timestamp)
        
        print(f"Results saved to {self.results_dir}/")
    
    def create_visualizations(self, timestamp):
        """Create and save visualizations of tournament results"""
        # Plot win rates
        plt.figure(figsize=(12, 6))
        self.stats['win_rate'].plot(kind='bar')
        plt.title('Win Rates by Player')
        plt.ylabel('Win Rate')
        plt.xlabel('Player')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/win_rates_{timestamp}.png")
        
        # Plot points
        plt.figure(figsize=(12, 6))
        self.stats['points'].plot(kind='bar')
        plt.title('Tournament Points by Player')
        plt.ylabel('Points')
        plt.xlabel('Player')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/points_{timestamp}.png")
        
        # Create heatmap of matchup results
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matchup_matrix, cmap='Blues')
        plt.colorbar(label='Wins')
        plt.title('Head-to-head Wins')
        plt.xticks(np.arange(len(self.player_ids)), self.player_ids, rotation=90)
        plt.yticks(np.arange(len(self.player_ids)), self.player_ids)
        
        # Add text annotations
        for i in range(len(self.player_ids)):
            for j in range(len(self.player_ids)):
                plt.text(j, i, f"{self.matchup_matrix.iloc[i, j]:.0f}", 
                        ha="center", va="center", color="white" if self.matchup_matrix.iloc[i, j] > 2 else "black")
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/matchup_heatmap_{timestamp}.png")
        
        # Close all figures to free memory
        plt.close('all')

if __name__ == "__main__":
    # Define players with different parameter configurations
    players = [        
        # MCTS with different simulation counts and exploration parameters
        # (MCTS, {"num_simulations": 64, "c": 2, "temp": 1}),
        # (MCTS, {"num_simulations": 256, "c": 1.4, "temp": 1}),
        (MCTS, {"num_simulations": 1024, "c": 1.4, "temp": 1}),
        
        (NNAI_greedy, {"model_path": "models_ppo/pretrained_ppo.pt"}),
        # (NNAI_greedy, {"model_path": "models_ppo/ppo_iter_240.pt"}),

        (NNAIv2, {"num_simulations": 1024, "c_puct": 0.75, "prob_temp": 0.8, "model_path": "models_ppo/ppo_iter_240.pt"}),
        # (NNAIv2, {"num_simulations": 16, "c_puct": 0.75, "prob_temp": 0.8, "model_path": "models_ppo/ppo_iter_240.pt"}),
        # (NNAIv2, {"num_simulations": 64, "c_puct": 1, "prob_temp": 1.2, "model_path": "models_ppo/ppo_iter_240.pt"}),

        # (NNAI, {"num_simulations": 64, "c_puct": 2, "prob_temp": 0.1, "model_path": "models_ppo/ppo_iter_240.pt"}),

    ]
    
    # Create and run tournament
    tournament = Tournament(
        players=players,
        games_per_matchup=20,  # Number of games each pair of players will play
        max_workers = 32,        # Adjust based on your system's capabilities
        max_moves=3000,        # Maximum moves per game before declaring a draw
        results_dir="tournament_results"
    )
    
    # Run the tournament
    results = tournament.run_tournament(save_results=True)
    
    # Display the top performers
    print("\nTop performers:")
    print(results.head())