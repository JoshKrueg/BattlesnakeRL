"""
Module: Snake Player with Visualization
Purpose: Game player class with Pygame visualization and AI opponent integration
Optimized: Extends GameState with graphics rendering
"""

import torch
import pygame
import sys
import ai_opponent as EnemyAI
import numpy as np
from game_state import GameState
from ai_neural_network import NNAI, NNAI_greedy
from ai_opponent import MCTS
from nn_mcts_search_v2 import NNAIv2


class GameStateVisualizer(GameState):
    def __init__(self, copy=None):
        super().__init__(copy)
        self.screen = pygame.display.set_mode((380, 380))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        # self.enemy_ai = NNAI_greedy(self, "models_ppo/ppo_iter_240.pt")
        # self.enemy_ai = NNAI_greedy(self, "models_ppo/pretrained_ppo.pt")
        self.enemy_ai = NNAIv2(self, num_simulations=1024, c_puct=0.8, prob_temp=0.8, model_path="models_ppo/ppo_iter_240.pt")
        #  self.enemy_ai = MCTS(self, num_simulations=2048, c=1.4, temp=1)


    def move(self, dir, snake):
        if snake == 1:
            self.flip_player()
            result = super().move(dir)
            self.flip_player()
        else:   
            result = super().move(dir)
        return result

    def draw(self):
        self.screen.fill((0, 0, 0))
        for y in range(1,20):
            for x in range(1,20):
                if self.gameboard[1, y, x] == 1:  # Food
                    color = (255, 0, 0)
                elif self.gameboard[2, y, x] > 0:  # Snake 0
                    color = (0, 255, 0)
                elif self.gameboard[4, y, x] > 0:  # Snake 1
                    color = (0, 0, 255)
                else:
                    continue
                pygame.draw.rect(self.screen, color, (x * 20 - 20, y * 20 - 20, 20, 20))
        pygame.display.flip()

    def gameplay_loop(self):
        self.setup()
        snake1_direction = np.array([0, 1])  # Snake 1 moves to the right
        directions = {
            pygame.K_UP: np.array([-1, 0], np.int16),
            pygame.K_DOWN: np.array([1, 0], np.int16),
            pygame.K_LEFT: np.array([0, -1], np.int16),
            pygame.K_RIGHT: np.array([0, 1], np.int16),
        }
        current_direction = np.array([0, 1], np.int16)  # Snake 0 starts moving to the right
        
        self.draw()

        while True:
            valid_dirs = self.get_valid_moves()
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in directions:
                    current_direction = directions[event.key]
                #     if not any([np.array_equal(current_direction, dir) for dir in valid_dirs]):
                #         continue
                # else:
                #     continue
            else:
                continue
            # Move Snake 0
            # current_direction = self.enemy_ai.get_move()
            result = self.move(current_direction, 0)
            if result == 1:  # Check for game over
                print("You Win!")
                pygame.quit()
                sys.exit()
            elif result == -1:  # Check for game over
                print("You Lose!")
                print(self)
                pygame.quit()
                sys.exit()

            # Move Snake 1
            self.flip_player()
            snake1_direction = self.enemy_ai.get_move()
            result = self.move(snake1_direction, 0)
            self.flip_player()
            print(self.snake0_health, self.snake1_health)
            if result == 1:  # Check for game over
                print("You Lose!")
                pygame.quit()
                sys.exit()
            elif result == -1:  # Check for game over
                print("You Win!")
                pygame.quit()
                sys.exit()

            # Draw the game state
            self.draw()


def main():
    pygame.init()
    game_state = GameStateVisualizer()
    game_state.gameplay_loop()

if __name__ == "__main__":
    main()