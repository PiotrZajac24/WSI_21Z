import matplotlib.pyplot as plt
import json
from itertools import combinations
from copy import deepcopy
from collections import defaultdict
import argparse
from pathlib import Path

def first_positive(l):
    for i, x in enumerate(l):
        if x > 0:
            return i
    return None

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="./results/qlearn_results.json", type=str)
    args = parser.parse_args()

    if Path(args.file).suffix != ".json":
        raise ValueError("Json file extension required.")
    return args


def plot_rewards_iteration(data):
    for k, v in data.items():
        r = list(range(1, len(v)+1))
        plt.title(f"Reward in each iteration, (alpha, gamma) = {k}")
        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.scatter(r, v)
        plt.show()
    return
    

if __name__ == "__main__":
    args = parse_arguments()
    try:
        data = {}
        with open(args.file) as f:
            data = json.load(f)
        plot_rewards_iteration(data)
    except Exception as e:
        print(e)
