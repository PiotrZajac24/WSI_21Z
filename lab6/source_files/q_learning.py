import numpy as np
import random
from collections import defaultdict
import json

class Board:

    def __init__(self, start, end, obstacle_ratio, n=8):
        self.n = n if 8 <= n < 100 else 8
        self.start = start if all(0 <= x < self.n for x in start) else (0, 0)
        self.end = end  if all(0 <= x < self.n for x in start) else (self.n-1, self.n-1)
        self.obstacle_ratio = obstacle_ratio if 0 <= obstacle_ratio <= 0.5 else 0.5
        self.rand_cell = np.vectorize(self._random_cell)
        self.actions = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
        self._generate_board()

    def _generate_board(self):
        # np.random.seed(100)
        for _ in range(1000):
            board = np.full((self.n, self.n), self.obstacle_ratio)
            board = self.rand_cell(board)
            board[self.start] = 0
            board[self.end] = 0
            if self._check_board(board, self.start, self.end):
                self.board = board
                self.board[self.start] = -1
                self.board[self.end] = 1000
                return board
        return None

    def _check_board(self, board, start, end):
        visited = np.full(board.shape, False)
        return self._search(np.array(start), np.array(end), board, visited)

    def _search(self, cell, goal, board, visited):
        if cell[0] == goal[0] and cell[1] == goal[1]:
            return True
        if board[cell[0], cell[1]] == -1000:
            visited[cell[0], cell[1]] = True
            return
        visited[cell[0], cell[1]] = True
        for action in self.actions:
            next_cell = cell + action
            if 0 <= next_cell[0] < self.n and  0 <= next_cell[1] < self.n: 
                if not visited[next_cell[0], next_cell[1]]:
                    result = self._search(next_cell, goal, board, visited)
                    if result:
                        return result
        return False

    def _random_cell(self, obstacle_ratio):
        return -1000 if np.random.uniform(0, 1) < obstacle_ratio else -1

    def __str__(self):
        return str(self.board)


class QLearning:
    
    def __init__(self, n, start, end, obstacle_ratio):
        self.start = start
        self.end = end
        self.n = n
        self.obstacle_ratio = obstacle_ratio
        self.board = Board(start, end, obstacle_ratio, n)
        self.actions = {i: k for i, k in enumerate(np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]))}
        self.q_table = np.zeros((self.n**2, len(self.actions)))
        self.rewards = np.array(
            [[self.board.board[i+x, j+y] if 0 <= i+x < self.n and 0 <= j+y < self.n else -1000 for x, y in self.actions.values()] \
            for i in range(self.n) for j in range(self.n)
        ])
        self.prev_action = {
            0: 3, 1: 2, 2: 1, 3: 0
        }
        self.iter_rewards = defaultdict(lambda: [])
        

    def learn(self, alpha, gamma, min_epsilon, max_epsilon, epochs, max_moves):
        self.q_table = np.zeros((self.n**2, len(self.actions)))
        e = 0
        epsilon = max_epsilon
        decay = 0.99
        while e < epochs:
            current_reward = 0
            i = 0
            state = self.start
            while i < max_moves:
                # choose random action or best action depending on drawn value
                if np.random.uniform(0, 1) <= epsilon:
                    action_ind = random.choice(list(self.actions.keys()))
                else:
                    actions = self.q_table[state[0]*self.n + state[1]]
                    action_ind = np.random.choice(np.flatnonzero(actions == max(actions)))
                i += 1

                reward = self.update_q(state, action_ind, alpha, gamma)
                if reward < -1:
                    break
                current_reward += reward
                if reward > 999:
                    break
                state = state + self.actions[action_ind]
                max_moves += 1
            epsilon = max(min_epsilon, epsilon*decay)   # decrease epsilon value by given ratio
            e += 1
            self.iter_rewards[str((alpha, gamma))].append(int(current_reward))
        return

    def update_q(self, state, action_ind, alpha, gamma):
        next_state = state + self.actions[action_ind]
        cell_index = state[0]*self.n+state[1]
        next_cell_index = next_state[0]*self.n+next_state[1]

        # check if next move leads out of map
        out_of_map = next_state[0] < 0 or next_state[0] >= self.n or next_state[1] < 0 or next_state[1] >= self.n
        reward = self.board.board[divmod(next_cell_index, self.n)] if not out_of_map else -1000

        self.q_table[cell_index, action_ind] = \
            (1 - alpha)*self.q_table[cell_index, action_ind] + \
                alpha*(reward + gamma * (max(self.q_table[next_cell_index]) if not out_of_map else -1000))

        return reward

    def move(self):
        # find best path depending on q-table values
        state = self.start
        moves = []
        action = None
        i = 0
        visited = np.full(self.board.board.shape, False)
        visited[state[0], state[1]] = True
        while i < self.n**2:
            curr_cell = state[0]*self.n + state[1]
            possible_actions = self.q_table[curr_cell]
            if action is not None:
                possible_actions[self.prev_action[action]] = -np.inf
            action = np.argmax(possible_actions)
            if possible_actions[action] == -np.inf:
                break
            state = state + self.actions[action]

            if self.board.board[divmod(curr_cell, self.n)] in (1000, -1000) or \
                state[0] < 0 or state[0] >= self.n or state[1] < 0 or state[1] >= self.n:
                break
            moves.append(self.actions[action])
            i += 1
        return moves

    
if __name__ == "__main__":
    n = 25
    start = (0, 0)
    end = (n-1, n-1)
    obstacle_ratio = 0.3
    QAgent = QLearning(n, start, end, obstacle_ratio)
    print(QAgent.board)

    for a in np.arange(0.1, 1.1, 0.1):
        for g in np.arange(0.1, 1.1, 0.1):
            print(round(a, 2), round(g, 2))
            QAgent.learn(round(a, 2), round(g, 2), 0.1, 1, 4000, n**2)
    
    with open("qlearn_results.json", "w+") as f:
        json.dump(QAgent.iter_rewards, f)
