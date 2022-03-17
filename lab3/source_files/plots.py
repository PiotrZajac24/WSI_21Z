import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file', type=str)  # name of file containing simulation results
    args = parser.parse_args()
    return args


def plot_time_depth(data):
    plt.figure()
    plt.title(f'Average time of minimax execution depending on depth.')
    plt.xlabel("Depth")
    plt.ylabel("Time [s]")

    time_results = data["time"]
    
    mean = np.array([np.mean(v) for _, v in time_results.items()])
    std = np.array([np.std(v) for _, v in time_results.items()])
    
    keys = [int(k) for k in time_results.keys()]
    plt.bar(keys, mean, yerr=std, capsize=16)
    plt.show()


def wins_after_first_move(data):
    results = data["wins_by_move"]

    wins = {}
    for k, v in results.items():
        res = Counter(x[0] for x in v)
        wins[int(k) // 6 + 1] = dict(sorted(res))
    
    for k, v in wins.items():
        print(k, v)
    return wins

    

def main():
    args = parse_arguments()
    try:
        with open(args.data_file, 'r') as f:
            data = json.load(f)
        plot_time_depth(data)
        wins_after_first_move(data)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
    