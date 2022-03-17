
import numpy as np

from source_files.connect_four import Connect4
from itertools import product
import json
from collections import Counter, defaultdict
import time

def random_games(board, games=40, verbose=False):
    results_random = []
    for _ in range(games):
        random_result = board.play_random(verbose)
        results_random.append(random_result)
    return results_random


def ai_games(board, depths, games=40, verbose=False):
    results_ai = []

    for _ in range(games):
        ai_result = board.play_ai(depths, verbose)
        results_ai.append(ai_result)
    return tuple(zip(*results_ai))

def random_ai_games(board, depth, ai_starts, games=20, verbose=False):
    results_random_ai = []
    for _ in range(games):
        random_ai_result = board.play_random_ai(depth, ai_starts, verbose)
        results_random_ai.append(random_ai_result)
    return results_random_ai


def main():
    games = 100
    board = Connect4(6, 7)
    min_depth, max_depth = 1, 6
    depths = list(product(list(range(min_depth, max_depth)), repeat=2))
    all_results = {
        "AI": defaultdict(lambda: {"wins": [], "avg_moves":[], "std_moves": []}), 
        "random": {}, "AI_random": defaultdict(lambda: {})}


    results = []
    for d in depths:
        t = time.perf_counter()
        res, boards = ai_games(board, d, games, verbose=False)
        print(time.perf_counter() - t)
        results.append((f"AI {d[0]} vs AI {d[1]}: ", Counter(res)))
        
        match_length = [board.size - state.count(' ') for state in boards]
        mean_moves = np.mean(match_length)
        std_moves = np.std(match_length)
        all_results["AI"][str(d)]["wins"] = dict(Counter(res))
        all_results["AI"][str(d)]["avg_moves"] = mean_moves
        all_results["AI"][str(d)]["std_moves"] = std_moves
        print(results[-1], mean_moves, std_moves)

    print()
    rand_games = Counter(random_games(board, 10*games))
    print("Random games: ", rand_games)
    all_results["random"] = dict(rand_games)

    for d in range(min_depth, max_depth-2):
        print("depth:", d)
        ai_rand = random_ai_games(board, d, True, games)
        print(f"AI {d} vs random: ", Counter(x[0] for x in ai_rand))
        all_results["AI_random"][str((d, 0))] = Counter(x[0] for x in ai_rand)
        rand_ai = random_ai_games(board, d, False,games)
        all_results["AI_random"][str((0, d))] = Counter(x[0] for x in rand_ai)
        print(f"random vs AI {d}: ", Counter(x[0] for x in rand_ai))

    all_results["time"] = board.time_results
    all_results["wins_by_move"] = board.wins_by_move 

    for r in results:
        print(r)

    with open('results.json', 'w') as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    main()