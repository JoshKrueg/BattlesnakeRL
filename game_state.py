"""
Module: Game State Management
Purpose: Core game state class for managing snake game board and player state
Optimized: Standard Python implementation with NumPy arrays
"""

import numpy as np

class GameState:
    def __init__(self, copy = None):
        #0 walls
        #1 food
        #2 snake0 body
        #3 snake0 head
        #4 snake1 body
        #5 snake1 head
        if copy == None:
            self.gameboard = np.zeros((6,21,21), dtype = np.int16)
            self.snake0_coords = None
            self.snake0_length = 3
            self.snake0_health = 70
            self.snake1_coords = None
            self.snake1_length = 3
            self.snake1_health = 70

            self.gameboard[0, 0, :] = 1
            self.gameboard[0,-1, :] = 1
            self.gameboard[0, :, 0] = 1
            self.gameboard[0, :,-1] = 1

        else:
            self.gameboard = copy.gameboard.copy()
            self.snake0_coords = copy.snake0_coords.copy()
            self.snake1_coords = copy.snake1_coords.copy()
            self.snake0_length = copy.snake0_length
            self.snake0_health = copy.snake0_health
            self.snake1_length = copy.snake1_length
            self.snake1_health = copy.snake1_health

    def setup(self):
        self.snake0_coords = np.array((7, 3), dtype = np.int16)
        self.snake1_coords = np.array((13, 3), dtype = np.int16)

        self.snake0_length = 3
        self.snake0_health = 70
        self.snake1_length = 3
        self.snake1_health = 70

        self.gameboard[1, :, :] = 0  # Clear any existing food
        self.gameboard[2, 7, 1:4] = np.array([1,2,3], dtype = np.int16)
        self.gameboard[3, 7, 3] = 70
        self.gameboard[4, 13, 1:4] = np.array([1,2,3], dtype = np.int16)
        self.gameboard[5, 13, 3] = 70

        self.add_food()

    def blocked_mask(self):
        return np.logical_or(self.gameboard[0], np.logical_or(self.gameboard[2], self.gameboard[4]))
        
    def flip_player(self):
        self.gameboard[2], self.gameboard[4] = self.gameboard[4].copy(), self.gameboard[2].copy()
        self.gameboard[3], self.gameboard[5] = self.gameboard[5].copy(), self.gameboard[3].copy()

        self.snake0_coords, self.snake1_coords = self.snake1_coords.copy(), self.snake0_coords.copy()
        self.snake0_length, self.snake1_length = self.snake1_length, self.snake0_length
        self.snake0_health, self.snake1_health = self.snake1_health, self.snake0_health

    def is_nonfatal(self, dir):
        unoccupied_square = self.blocked_mask()[*(self.snake0_coords + dir)].item() == 0
        winning_collision = (self.snake0_coords + dir == self.snake1_coords).all() and self.snake0_length > self.snake1_length
        return unoccupied_square or winning_collision

    def get_valid_moves(self):
        directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype = np.int16)
        return list(filter(self.is_nonfatal, directions))

    def add_food(self):
        valid_coords = np.nonzero(np.logical_not(np.logical_or(self.blocked_mask(), self.gameboard[1])))
        idx = np.random.randint(0, len(valid_coords[0]))
        new_food_coords = np.array((valid_coords[0][idx], valid_coords[1][idx]), dtype = np.int16)

        self.gameboard[1, *new_food_coords] = 1

    def rand_add_food(self):
        threshold = 0.05 /  (2 ** np.sum(self.gameboard[1] == 1))

        if np.random.random() > threshold:
            return

        valid_coords = np.nonzero(np.logical_not(np.logical_or(self.blocked_mask(), self.gameboard[1])))
        idx = np.random.randint(0, len(valid_coords[0]))
        new_food_coords = np.array((valid_coords[0][idx], valid_coords[1][idx]), dtype = np.int16)

        self.gameboard[1, *new_food_coords] = 1

    def move_no_new_food(self, dir):
        self.gameboard[3, *self.snake0_coords] = 0
        self.snake0_coords += dir

        if (self.snake0_coords == self.snake1_coords).all():
            return 2 * int(self.snake0_length > self.snake1_length) - 1

        elif self.gameboard[1, *self.snake0_coords] == 1:
            self.snake0_health = 70
            self.snake0_length += 1
            self.gameboard[1, *self.snake0_coords] = 0  # No new food added
        
        elif self.snake0_health == 0:
            return -1

        else:
            self.snake0_health -= 1
            self.gameboard[2] = np.maximum(self.gameboard[2] - 1, np.array(0))

        self.gameboard[3, *self.snake0_coords] = self.snake0_health
        result = -int(self.blocked_mask()[*self.snake0_coords])
        self.gameboard[2, *self.snake0_coords] = self.snake0_length

        return result
    
    def move(self, dir):
        self.gameboard[3, *self.snake0_coords] = 0
        self.snake0_coords += dir

        if (self.snake0_coords == self.snake1_coords).all():
            return 2 * int(self.snake0_length > self.snake1_length) - 1

        elif self.gameboard[1, *self.snake0_coords] == 1:
            self.snake0_health = 70
            self.snake0_length += 1
            self.gameboard[3, *self.snake0_coords] = self.snake0_health
            self.gameboard[2, *self.snake0_coords] = self.snake0_length
            self.gameboard[1, *self.snake0_coords] = 0
            self.rand_add_food()
            return 0
        
        elif self.snake0_health == 0:
            return -1

        else:
            self.snake0_health -= 1
            self.gameboard[2] = np.maximum(self.gameboard[2] - 1, np.array(0))
            self.gameboard[3, *self.snake0_coords] = self.snake0_health
        
        result = -int(self.blocked_mask()[*self.snake0_coords])
        self.gameboard[2, *self.snake0_coords] = self.snake0_length

        self.rand_add_food()

        return result
    
    def get_next_state(self, dir):
        next_state = GameState(self)
        r = next_state.move_no_new_food(dir)
        return next_state, r
    
    def to_tuple(self):
        return (
            self.gameboard.copy(),
            self.snake0_coords.copy(),
            self.snake0_length,
            self.snake0_health,
            self.snake1_coords.copy(),
            self.snake1_length,
            self.snake1_health
        )

    def __repr__(self):
        board_repr = ""
        for y in range(1, self.gameboard.shape[1]-1):
            for x in range(1, self.gameboard.shape[2]-1):
                if self.gameboard[1, y, x] == 1:  # Food
                    board_repr += "F"
                elif self.gameboard[2, y, x] > 0:  # Snake 0 (head or body)
                    board_repr += "O"
                elif self.gameboard[4, y, x] > 0:  # Snake 1 (head or body)
                    board_repr += "X"
                else:
                    board_repr += " "  # Empty space
            board_repr += "\n"
        return board_repr


