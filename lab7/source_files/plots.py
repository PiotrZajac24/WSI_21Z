import matplotlib.pyplot as plt
import json
from itertools import combinations
from copy import deepcopy
from collections import defaultdict
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="./results/results_naive_bayes.json", type=str)
    args = parser.parse_args()

    if Path(args.file).suffix != ".json":
        raise ValueError("Json file extension required.")
    return args


def plot_classes_by_attribute_values(df):
    classes = sorted(list(set(df[df.columns[-1]])))
    filters = list(combinations(df.columns[:-1], 2))

    for filter in filters:
        plt.title("Distribution of classes depending on attributes: " + ", ".join(filter))
        plt.xlabel(filter[0])
        plt.ylabel(filter[1])
        
        sample_data = deepcopy(df)
        for class_ in classes:
            class_occ = sample_data[sample_data[df.columns[-1]] == class_]
            x = class_occ[filter[0]]
            y = class_occ[filter[1]]
            plt.scatter(x, y, label=class_)

        plt.legend()
        plt.show()
    return
    

def get_accuracies(data):
    return [v["accuracy"] for k, v in data.items()]


def plot_accuracy(file):
    with open(file) as f:
        data = json.load(f)

    sorted_results = data["sorted"]
    shuffled_results = data["random"]
    k = list(map(int, sorted_results.keys()))

    plt.figure().clear()
    plt.title("Accuracy depending on k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")

    plt.plot(k, get_accuracies(sorted_results), label="sorted")
    plt.plot(k, get_accuracies(shuffled_results), label="shuffled")
    plt.legend()
    plt.show()


def plot_metrics(file, key):
    with open(file) as f:
        data = json.load(f)[key]

    k_list = list(map(int, data.keys()))
    results_classes = defaultdict(lambda: defaultdict(lambda: []))

    for _, v in data.items():
        for metric in ("recall", "precision"):
            for _class, res in v[metric].items():
                results_classes[metric][_class].append(res)

    for m, res in results_classes.items():
        plt.figure().clear()
        plt.title(f"{m.title()} values depending on k for each class ({key} data)")
        plt.xlabel("k")
        plt.ylabel(m.title())
        for _class, k_val in res.items():
            plt.plot(k_list, k_val, label=_class)
        plt.legend()
        plt.show()
    

if __name__ == "__main__":
    args = parse_arguments()
    try:
        plot_accuracy(args.file)
        plot_metrics(args.file, "sorted")
        plot_metrics(args.file, "random")
    except Exception as e:
        print(e)


