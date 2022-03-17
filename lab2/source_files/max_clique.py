# WSI LAB 2 - Piotr Zając

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
from functools import lru_cache
from collections import Counter
from itertools import combinations
import json
from pathlib import Path


def unique(lst):
    return sorted(list(set(lst)))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nodes', nargs='+', default=[50], type=int)            # amount of edges in a graph
    parser.add_argument('--edge_node_ratio', nargs='+',default=[0.4], type=float)   # percent of edges to remain in a graph
    parser.add_argument('--gen_mutations', nargs='+',default=[1], type=int)     # gene mutatation probability = gen_mutations/nodes 
    parser.add_argument('--size', nargs='+', default=[20, 50, 100], type=int)             # size of population
    parser.add_argument('--alg_runs', default=25, type=int)         # amount of genetic algorithm repetitions for each set of parameters
    parser.add_argument('--generations', default=500, type=int)     # stop condition
    parser.add_argument('--save_results', default=True, type=bool) # save results to json file
    parser.add_argument('--file', default="results.json", type=str) # specify the name of output file
    args = parser.parse_args()
    args.chrom_prob = [round(i*0.05, 2) for i in range(1, 20)]
    args.nodes = unique(args.nodes)
    if any(n <= 2 for n in args.nodes):
        raise ValueError("There should be at least two nodes")
    args.size = unique(args.size)
    if any(s <= 0 for s in args.size):
        raise ValueError("There should be at least one entity.")
    args.edge_node_ratio = unique(args.edge_node_ratio)
    if any(not 0 < r <= 1 for r in args.edge_node_ratio):
        raise ValueError("Edge ratio should fall within range between 0 and 1.")
    if args.generations <= 1:
        raise ValueError("There should be at least two generations.")
    if Path(args.file).suffix != ".json":
        raise ValueError("Json file extension required.")
    args.chrom_prob = unique(args.chrom_prob)
    if any(not 0 <= cp <= 1 for cp in args.chrom_prob):
        raise ValueError("Chromosome mutatation prob. should fall within range between 0 and 1.")
    args.gen_mutations = unique(args.gen_mutations)
    if any(0 > pg or min(args.nodes) < pg for pg in args.gen_mutations):
        print(args.nodes, args.gen_mutations)
        raise ValueError(f"1 to {args.nodes} might mutate.")
    return args


def timer(func):
    # decorator to measure time of function execution
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        stop = round(time.perf_counter() - start, 2)
        return result, stop
    return wrapper


def swap(binary):
    if binary not in '01':
        raise ValueError
    return '0' if binary == '1' else '1'


def random_graph(nodes=20, edge_ratio=0.4):
    random.seed(123279) # uncomment this line to generate same graphs as in report
    all_nodes = list(range(nodes))
    all_edges = list(combinations(all_nodes, 2))
    graph = nx.Graph()  # initialize graph
    graph.add_nodes_from(all_nodes) # add all nodes
    random.shuffle(all_edges) 
    graph.add_edges_from(all_edges[:int(edge_ratio*len(all_edges))])  # add amount of edges proportional to edge ratio
    return graph


def generate_population(size, genes, graph, initial_edges=True):
    # generate completely random population or use graph edges
    if initial_edges:
        edges = list(graph.edges())
        random.shuffle(edges)
        return [''.join('0' if i not in edge else '1' for i in range(genes)) for edge in edges[:min(len(edges),size)]]
    else:
        return [''.join(random.choice('01') for _ in range(genes)) for _ in range(size)]


@timer
def genetic_algorithm(graph, prob_chrom, prob_gen, max_iter, population):
    iter = 0
    chrom, rating = rate(population, graph)
    last_change = 0
    while iter < max_iter:
        population = tournament_selection(population, graph)  
        population = mutations(population, prob_chrom, prob_gen)
        new_chrom, new_rating = rate(population, graph)     # best member of new population and its rating
        if rating < new_rating:
            chrom, rating = new_chrom, new_rating
            last_change = iter
        iter += 1
    return chrom, last_change


def rate(population, graph):
    # rate each member of population by using evaluation function and return the best one
    best_chrom, max_clique = [], -1
    for chrom in population:
        last_rating = q(chrom, graph)
        if  last_rating > max_clique:
            max_clique = last_rating
            best_chrom = chrom
    return best_chrom, max_clique


def tournament_selection(population, graph, k=2):
    # select n groups of k members, and choose the best one from each group to generate new population (n - size of population)
    new_population = []
    for _ in population:
        candidates = random.sample(population, k)
        candidates.sort(key=lambda o: q(o, graph))
        new_population.append(candidates[-1])
    return new_population


def mutations(population, prob_chrom, prob_gen):
    # perform mutations on population, depending on chromosome and gene mutation probabilities
    for i, chrom in enumerate(population):
        if random.uniform(0, 1) <= prob_chrom:
            population[i] = ''.join(swap(c) if random.uniform(0, 1) <= prob_gen else c for c in chrom)
    return population


@lru_cache
def q(chrom, graph):
    # evaluation function
    # returns k if subgraph is a clique (k - amount of nodes in a subgraph) or 0 otherwise
    nodes = {i for i, v in enumerate(chrom) if v == '1'}
    k = len(nodes)
    for i in nodes:
        if len(set(graph.neighbors(i)).intersection(nodes)) != k-1:
            return 0
    return k if k != 1 else 0


def plot_result_graph(graph, chrom):
    color_map = []
    for x in chrom:
        color_map.append('red' if x == '1' else 'gray')
    nx.drawing.nx_pylab.draw_circular(graph, node_color=color_map, with_labels=True)

    nodes = {i for i, v in enumerate(chrom) if v == '1'}

    temp = {}
    for i in nodes:
        temp[i] = set(graph.neighbors(i)).intersection(nodes)
    
    edges = []
    for k, v in temp.items():
        edges.extend([(k, v2) for v2 in v])
    edges = list(set([(min(x), max(x)) for x in edges]))

    nx.draw_networkx_edges(graph,  pos=nx.circular_layout(graph), edgelist=edges, edge_color='red')
    plt.show()
    #plt.savefig('graph.png')


def main():
    args = parse_arguments()
    print(args)
    results = []
    for nodes in args.nodes:
        for edge_ratio in args.edge_node_ratio:
            graph = random_graph(nodes, edge_ratio) # generate graph
            random.seed()   # set random seed
            for size in args.size:
                for chrom_prob in args.chrom_prob:
                    for gen_mutations in args.gen_mutations:
                        found_cliques, last_changes, time_results = [], [], []

                        for _ in range(args.alg_runs):
                            population = generate_population(size, len(graph.nodes()), graph)
                                
                            (result, last_change), t = genetic_algorithm(
                                graph, chrom_prob,
                                gen_mutations/nodes,
                                args.generations, population
                            )
                            #plot_result_graph(graph, result)
                            found_cliques.append(result)
                            last_changes.append(last_change)
                            time_results.append(t)
                        print(20*"_")
                        print(f"Nodes: {nodes}; Percent of edges: {edge_ratio}; Chromosome mut. prob.: {chrom_prob}; Gene mut. prob.{round(gen_mutations/nodes, 2)}; Population size: {size}")
                        n_cliques = [q(r, graph) for r in found_cliques]
                        print("Cliques of different sizes: ", Counter(n_cliques))
                        print("Standard deviation (clique size)", round(np.std(n_cliques), 2))
                        print("Average clique: ", round(np.mean(n_cliques), 2))
                        print("Average time: ", round(np.mean(time_results), 2))
                        print("Standard deviation (time):", round(np.std(time_results), 2))
                        print("Average iterations to reach max clique ", round(np.mean(last_changes), 2))
                        print("Standard deviation (last iteration with larger clique)", round(np.std(last_changes), 2))


                        if args.save_results:
                            results.append({
                                'nodes': nodes,
                                'edge_node_ratio': edge_ratio,
                                'population_size': size,
                                'chrom_mutation': chrom_prob,
                                'gen_mutations': gen_mutations,
                                'gen_prob': gen_mutations/nodes,
                                'generations': args.generations,
                                'mean_time': np.mean(time_results),
                                'std_time': np.std(time_results),
                                'avg_changes': np.mean(last_changes),
                                'std_changes': np.std(last_changes),
                                'avg_clique': np.mean(n_cliques),
                                'std_clique': np.std(n_cliques),
                                'max_clique': max(n_cliques),
                                'min_clique': min(n_cliques)
                            })
    if args.save_results:
        with open(args.file, 'w') as f:
            json.dump(results, f)
    

if __name__ == "__main__":
    main()
