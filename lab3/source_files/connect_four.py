# WSI LAB 3 - Piotr Zaj�c

import argparse
import numpy as np
import random
from math import inf
from collections import Counter, defaultdict
from copy import deepcopy
import time


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--width', default=7, type=int)
    parser.add_argument('--height', default=6, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--games', default=1, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    args = parser.parse_args()

    if args.width < 5:
        raise ValueError("Width should be higher than 4.")
    if args.height < 4:
        raise ValueError("Height should be higher than 3.")
    if args.depth < 1:
        raise ValueError("Height should be higher than 0.")    
    return args


class Connect4:
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.size = self.height*self.width

        self._all_fours = self.fours()
        self._remaining_fours = deepcopy(self._all_fours)
        self._cells = self.four_by_cells(self._all_fours)
        self._threat_rate = {4: inf, 3:10, 2: 4, 1:0}

        self._state = " "*self.height*self.width
        self._max_move = True
        self._colors = {True: "Y", False: "R"}
        self.time_results = defaultdict(lambda: [])
        self.wins_by_move = defaultdict(lambda: [])

    def _reset_board(self):
        self._remaining_fours = deepcopy(self._all_fours)
        self._state = " "*self.height*self.width
        self._max_move = True

    def fours(self):
        # generate all lines consisting of 4 cells
        h = self.height
        w = self.width
        vertical = np.array([0, 1, 2, 3])
        horizontal = np.array([0, h, 2*h, 3*h])
        down_up = np.array([0, h+1, 2*(h+1), 3*(h+1)])
        up_down = np.array([0, h-1, 2*(h-1), 3*(h-1)])
        all_fours, size = [], h*w

        for i in range(size):
            pos = vertical + i
            if len(set(x//h for x in pos)) == 1:
                all_fours.append(pos)

            pos = horizontal + i
            if all(x < size for x in pos):
                all_fours.append(pos)

            pos = down_up + i
            diff = [x//h for x in pos]
            if all(x - diff[i-1] == 1 for i, x in enumerate(diff[1:], 1)) and all(x < size for x in pos):
                all_fours.append(pos)

            pos = up_down + i
            diff = [x//h for x in pos]
            if all(x - diff[i-1] == 1 for i, x in enumerate(diff[1:], 1)) and all(x < size for x in pos):
                all_fours.append(pos)
        return tuple(tuple(four) for four in all_fours)

    def four_by_cells(self, fours):
        # generate all threes neighbouring to given cell
        cells = {i:{} for i in range(self.height*self.width)}
        for k in cells.keys():
            cells[k][False] = [[j for j in line if k != j] for line in fours if k in line]
        for k in cells.keys():
            cells[k][True] = [[j for j in line if k+1 != j] for line in cells[k+1][False] 
                if k not in line] if k % self.height+1 != self.height else []
        return cells

    def print_board(self, state):
        board = ""
        for i in range(self.height-1, -1, -1):
        	board += '|'.join(state[j*self.height+i]for j in range(self.width)) + "\n"
        print(board)
        return board

    def check_win(self, state, last_index):
        # check neighbours of cell indicated by last move index to find four tokens in a row
        if last_index < 0:
            return False
        for line in self._cells[last_index][False]:
            if all(state[c] == state[last_index] for c in line):
                return state[last_index]
        if state.count(' ') == 0:
            return "D"
        return False

    def available_moves(self, state):
        # find all available moves
        moves = []
        for i in range(self.width):
            tmp = i*self.height
            if state[tmp+self.height-1] == ' ':
                moves.append(tmp+state[tmp:tmp+self.height].index(' '))
        return moves

    def heuristic(self, state, last_index, max_move):
        # check if position is terminal - if not count threats of both players and count difference
        result = self.check_win(state, last_index)
        if not result:
            points = {"Y": 0, "R": 0}
            defaultdict(lambda: 0)
            curr_turn, next_turn = self._colors[max_move], self._colors[not max_move]

            for four in self._remaining_fours:
                cells = Counter([state[i] for i in four])
                if curr_turn in cells and next_turn in cells:
                    continue
                elif curr_turn in cells:
                    points[curr_turn] += (self._threat_rate[cells[curr_turn]])
                elif next_turn in cells:
                    points[next_turn] += (self._threat_rate[cells[next_turn]])
            
            return points["Y"] - points["R"]
        else:
            return inf if result == "Y" else -inf if result == "R" else 0

    def next_states(self, state, successors, max_move):
        # generate all following positions
        return [(state[:move] + self._colors[max_move] + 
            (state[move+1:] if move + 1 < self.size else '')
            , move)
             for move in successors]

    def new_state(self, state, move, max_move):
        return state[:move] + self._colors[max_move] + (state[move+1:] 
         if move + 1 < self.size else '')

    def minimax(self, state, depth, max_move, alpha, beta, last_move):
        # check termination conoditions
        if depth == 0 or self.check_win(state, last_move):
            return self.heuristic(state, last_move, max_move), state, last_move

        successors = self.available_moves(state)
        if not successors:
            return 0, state, last_move
        # ggenerate all following positions
        U = self.next_states(state, successors, max_move)

        w = defaultdict(lambda: [])

        # for each following position call minimax function and add result to all
        for u, move in U:
            score, _, _ = self.minimax(u, depth-1, not max_move, alpha, beta, move)
            w[score].append((u, move))

        # find min/max rating and choose one of moves with the same rating
        if max_move:
            mx = max(w.keys())
            next_state, move = random.choice(w[mx])
            return mx, next_state, move
        else:
            mn = min(w.keys())
            next_state, move = random.choice(w[mn])
            return mn, next_state, move

    def alpha_beta(self, state, depth, max_move, alpha, beta, last_move):
        if depth == 0 or self.check_win(state, last_move):
            return self.heuristic(state, last_move, max_move), state, last_move

        successors = self.available_moves(state)
        if not successors:
            return 0, state, last_move
        random.shuffle(successors)

        U = self.next_states(state, successors, max_move)
        w = defaultdict(lambda: [])

        if max_move:
            for u, move in U:
                score, _, _ = self.alpha_beta(u, depth-1, not max_move,
                 alpha, beta, move)
                w[score].append((u, move))
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            mx = max(w.keys())
            next_state, best_move = random.choice(w[mx])
            return alpha, next_state, best_move
        else:
            for u, move in U:
                score, _, _ = self.alpha_beta(u, depth-1, not max_move,
                 alpha, beta, move)
                w[score].append((u, move))
                beta = min(beta, score)
                if alpha >= beta:
                    break
            mn = min(w.keys())
            next_state, best_move = random.choice(w[mn])
            return beta, next_state, best_move

    def move(self, state, depth, max_move, last_move):
        successors = self.available_moves(state)
        curr_turn = self._colors[max_move]
        next_turn = self._colors[not max_move]

        # look for any winning move / opponent's winning move
        defeat = []
        for move in successors:
            for line in self._cells[move][False]:
                c = Counter([state[x] for x in line])
                if c[curr_turn] == 3:
                    return inf if curr_turn == "Y" else -inf, self.new_state(state, move, max_move), move
                elif c[next_turn] == 3:
                    defeat.append(move)
        if defeat:
            move = random.choice(defeat)
            new_state = self.new_state(state, move,  max_move)
            return self.heuristic(new_state, move, not max_move), self.new_state(state, move,  max_move), move 
        
        
        t = time.perf_counter()
        # use minimax to find optimal move

        # score, self._state, last_move = self.alpha_beta(state, depth, max_move, -inf, inf, last_move)
        score, self._state, last_move = self.minimax(state, depth, max_move, -inf, inf, last_move)
        
        self.time_results[depth].append(time.perf_counter() - t)

        # reduce amount of possible fours
        fours_left = []
        for four in self._remaining_fours:
            c = Counter(self._state[x] for x in four)
            if ("Y" in c and "R" in c) or ' ' not in c:
                pass
            else:
                fours_left.append(four)
        
        self._remaining_fours = fours_left
        return score, self._state, last_move
    
    def random_move(self, state, max_move):
        # play random move
        possible_moves = self.available_moves(state)
        if not possible_moves:
            return 0, state, max_move
        move = random.choice(possible_moves)
        new_state = self.new_state(state, move, max_move)
        score = self.heuristic(new_state, move, max_move)
        return score, new_state, move

    def play_random(self, verbose=False):
        self._reset_board()
        last_move = -1
        
        _, self._state, last_move = self.random_move(self._state, self._max_move)
        first_move = last_move
        self._max_move = not self._max_move

        i = 1
        while not self.check_win(self._state, last_move) and i < self.size:
            _, self._state, last_move = self.random_move(self._state, self._max_move)
            self._max_move = not self._max_move
            i += 1
        result = self.check_win(self._state, last_move)
        
        if verbose:
                print("____________")
                print("____________")
                print("FINISHED")
                print(result, " wins", last_move)
                print(20*"*")
                self.print_board(self._state)
                print(20*"*")
        
        result = result if result else "D"
        self.wins_by_move[first_move].append((result, self._state))
        return result

    def play_random_ai(self, depth, ai_starts=True, verbose=False):
        self._reset_board()
        last_move = -1

        random_turn = not ai_starts

        if random_turn:
                _, self._state, last_move = self.random_move(self._state, self._max_move)
        else:
            score, self._state, last_move = self.move(
                self._state, depth,
                self._max_move, last_move
            )
        first_move = last_move
        random_turn = not random_turn
        self._max_move = not self._max_move

        i = 1
        while not self.check_win(self._state, last_move) and i < self.size:
            if random_turn:
                _, self._state, last_move = self.random_move(self._state, self._max_move)
            else:
                score, self._state, last_move = self.move(
                    self._state, depth,
                    self._max_move, last_move
                )

            if verbose:
                print(20*"*")
                print(last_move, "Y" if self._max_move else "R", score)
                print(self._state,'!!!')
                self.print_board(self._state)
                print(20*"_")
            self._max_move = not self._max_move
            random_turn = not random_turn
            i += 1
        result = self.check_win(self._state, last_move)
        
        if verbose:
                print("____________")
                print("____________")
                print("FINISHED")
                print(result, " wins", last_move)
                print(20*"*")
                self.print_board(self._state)
                print(20*"*")
        
        result = result if result else "D"
        self.wins_by_move[first_move].append((result, self._state))

        return result, self._state

    def play_ai(self, depths, verbose=False):
        self._reset_board()
        last_move = -1
        
        player_depth = {True: depths[0], False: depths[1]}

        score, self._state, last_move = self.move(
                self._state, player_depth[self._max_move],
                 self._max_move, last_move
            )
        first_move = last_move
        self._max_move = not self._max_move
        i = 1
        while not self.check_win(self._state, last_move) and i < self.size:
            score, self._state, last_move = self.move(
                self._state, player_depth[self._max_move],
                 self._max_move, last_move
            )
            
            if verbose:
                print(20*"*")
                print(last_move, "Y" if self._max_move else "R", score)
                print(self._state,'!!!')
                self.print_board(self._state)
                print(20*"_")
            self._max_move = not self._max_move
            i += 1
        result = self.check_win(self._state, last_move)
        
        if verbose:
                print("____________")
                print("____________")
                print("FINISHED")
                print(result, " wins", last_move)
                print(20*"*")
                self.print_board(self._state)
                print(20*"*")
            
        result = result if result else "D"
        self.wins_by_move[first_move].append((result, self._state))
        return result, self._state

def main():
    args = parse_arguments()
    game = Connect4(args.height, args.width)

    results = []
    winners = []
    states = []
    for i in range(args.games):
        print("GAME ", i+1)
        r = game.play_ai([args.depth, args.depth], True)
        results.append(r)
        winners.append(r[0])
        states.append(results[-1][1])
        print(len(set(states)))
        print(Counter(winners))
        print(results[-1][0], " wins")
    
    winners = [a[0] for a in results]
    states = [a[1] for a in results]
    print(Counter(winners))
    print(len(set(states)))
    
if __name__ == "__main__":
    main()