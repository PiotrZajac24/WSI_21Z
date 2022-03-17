import matplotlib.pyplot as plt
from collections import defaultdict
import json
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_file', type=str)  # name of file containing simulation results
    # parameters used for grouping results
    parser.add_argument('--nodes', default=30, type=int)
    parser.add_argument('--edge_node_ratio', default=0.4, type=float)
    parser.add_argument('--gen_mutations', default=1, type=int)
    args = parser.parse_args()

    if args.nodes <= 0:
        raise ValueError("There should be at least one node.")
    if not 0 <= args.edge_node_ratio <= 1:
        raise ValueError("Edge ratio should fall within range between 0 and 1.")
    if args.gen_mutations <= 0:
        raise ValueError("There should be at least two generations.")
    return args


def plot_changes_chrom_prob(data, nodes, edge_node_ratio, gen_mutations, std=True, chrom_prob=[round(0.05*i, 20) for i in range(1, 20)]):
    plt.figure()
    plt.title(f'Average iterations to reach max found clique depending on chrom. mut. prob.\n  and population size ({nodes} nodes, {edge_node_ratio*100}% edges, gene mut. prob.  - {gen_mutations}/nodes)')
    plt.grid()
    plt.xlabel("Chromosome mutation probability")
    plt.ylabel("Iterations")
    time_edges = defaultdict(lambda: [])
    mx = 0
    for sample in data:
        time_edges[
            sample["nodes"],
            sample["edge_node_ratio"], 
            sample["gen_mutations"],
            sample["population_size"]
        ].append((sample["avg_changes"], sample["std_changes"]))
        mx = max(sample["avg_changes"], sample["std_changes"], mx)
    plt.gca().set_ylim(bottom=0, top=mx)
    for k, v in time_edges.items():
        mean_std = list(zip(*v))
        mean, std = np.array(mean_std[0]), np.array(mean_std[1])
        if k[0] == nodes and k[1] == edge_node_ratio and k[2] == gen_mutations:
            plt.plot(chrom_prob, mean, label=f"{k[-1]} chromosomes")
            plt.fill_between(chrom_prob, mean-std, mean+std, alpha=0.1)
    plt.legend()
    plt.show()

def plot_time_chrom_prob(data, nodes, edge_node_ratio, gen_mutations, std=True, chrom_prob=[round(0.05*i, 2) for i in range(1, 20)]):
    plt.figure()
    plt.title(f'Average time of function execution depending on chrom. mut. prob.\n  and population size ({nodes} nodes, {edge_node_ratio*100}% edges, gene mut. prob. - {gen_mutations}/nodes)')
    plt.grid()
    plt.xlabel("Chromosome mutation probability")
    plt.ylabel("Time [s]")

    time_edges = defaultdict(lambda: [])
    for sample in data:
        time_edges[
            sample["nodes"],
            sample["edge_node_ratio"], 
            sample["gen_mutations"],
            sample["population_size"]
        ].append((sample["mean_time"], sample["std_time"]))
    
    for k, v in time_edges.items():
        mean_std = list(zip(*v))
        mean, std = np.array(mean_std[0]), np.array(mean_std[1])
        if k[0] == nodes and k[1] == edge_node_ratio and k[2] == gen_mutations:
            plt.plot(chrom_prob, mean, label=f"{k[-1]} chromosomes")
            plt.fill_between(chrom_prob, mean-std, mean+std, alpha=0.1)
    plt.legend()
    plt.show()

def plot_clique_chrom_prob(data, nodes, edge_node_ratio, gen_mutations, std=True, chrom_prob=[round(0.05*i, 2) for i in range(1, 20)]):
    plt.figure()
    plt.title(f'Average size of found clique depending on chrom. mut. prob.\n  and population size ({nodes} nodes, {edge_node_ratio*100}% edges, gene mut. prob.  - {gen_mutations}/nodes)')
    plt.grid()
    plt.xlabel("Chromosome mutation probability")
    plt.ylabel("Max clique")
    node_edges = defaultdict(lambda: [])
    for sample in data:
        node_edges[
            sample["nodes"],
            sample["edge_node_ratio"], 
            sample["gen_mutations"],
            sample["population_size"]
        ].append((sample["avg_clique"], sample["std_clique"]))

    for k, v in node_edges.items():
        mean_std = list(zip(*v))
        mean, std = np.array(mean_std[0]), np.array(mean_std[1])
        if k[0] == nodes and k[1] == edge_node_ratio and k[2] == gen_mutations:
            plt.plot(chrom_prob, mean, label=f"{k[-1]} chromosomes")
            plt.fill_between(chrom_prob, mean-std, mean+std, alpha=0.1)
    plt.legend()
    plt.show()
    

def main():
    args = parse_arguments()
    try:
        with open(args.data_file, 'r') as f:
            data = json.load(f)
        any_data = False
        for sample in data:
            if sample['nodes'] == args.nodes and sample['edge_node_ratio'] == args.edge_node_ratio and sample['gen_mutations'] == args.gen_mutations:
                any_data = True
                break
        if not any_data:
            print("Not enough data for given parameters")
            return
        plot_clique_chrom_prob(data, args.nodes, args.edge_node_ratio, args.gen_mutations)
        plot_time_chrom_prob(data, args.nodes, args.edge_node_ratio, args.gen_mutations)
        plot_changes_chrom_prob(data, args.nodes, args.edge_node_ratio, args.gen_mutations)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
    