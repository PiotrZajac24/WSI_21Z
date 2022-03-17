import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from math import inf, sqrt


def magnitude(vector):
    return sqrt(sum(a**2 for a in vector))


def plot_iterations(all_results):
    plt.grid()
    plt.xlim(0, 1)
    plt.title("Average number of iterations")
    plt.yscale('log')
    plt.xlabel('beta')
    plt.ylabel('average iterations')

    grad_iter = defaultdict(lambda: [])
    newton_iter = defaultdict(lambda: [])
    for b, points in all_results.items():
        for _, results in points.items():
            if results["gradient"]["x"] != inf and results["gradient"]["y"] != inf:
                grad_iter[b].append(results["gradient"]["iterations"])
            if results["newton"]["x"] != inf and results["newton"]["y"] != inf:
                newton_iter[b].append(results["newton"]["iterations"])

    plt.scatter(list(grad_iter.keys()), list(sum(i)//len(i) for i in grad_iter.values()), label="Gradient")
    plt.scatter(list(newton_iter.keys()), list(sum(i)//len(i) for i in newton_iter.values()), label="Newton's method")
    plt.legend()
    plt.show()

def plot_time(all_results):
    plt.grid()
    plt.axis([0, 1, 0, 1.5])
    plt.title("Average time of function execution")
    plt.xlabel('beta')
    plt.ylabel('time')
    grad_time = defaultdict(lambda: [])
    newton_time = defaultdict(lambda: [])
    for b, points in all_results.items():
        for _, results in points.items():
            if results["gradient"]["x"] != inf and results["gradient"]["y"] != inf:
                grad_time[b].append(results["gradient"]["time"])
            if results["newton"]["x"] != inf and results["newton"]["y"] != inf:
                newton_time[b].append(results["newton"]["time"])

    plt.scatter(list(grad_time.keys()), list(sum(i)/len(i) for i in grad_time.values()), label="Gradient")
    plt.scatter(list(newton_time.keys()), list(sum(i)/len(i) for i in newton_time.values()), label="Newton's method")
    plt.legend()
    plt.show()

def plot_time_iterations(all_results):
    plt.grid()
    plt.axis([0, 30000, 0, 5])
    plt.title("Comparision between iterations and time")
    plt.xlabel('iterations')
    plt.ylabel('time')
    grad = []
    newton = []
    for _, points in all_results.items():
        for _, results in points.items():
            if results["gradient"]["x"] != inf and results["gradient"]["y"] != inf:
                grad.append((results["gradient"]["iterations"], results["gradient"]["time"]))
            if results["newton"]["x"] != inf and results["newton"]["y"] != inf:
                newton.append((results["newton"]["iterations"], results["newton"]["time"]))

    plt.scatter([val[0] for val in grad], [val[1] for val in grad], label="Gradient")
    plt.scatter([val[0] for val in newton], [val[1] for val in newton], label="Newton's method")
    plt.legend()
    plt.show()

def plot_accuracy(all_results):
    plt.grid()
    plt.xlim(0, 1)
    plt.title("Average distance from minimum")
    plt.xlabel('beta')
    plt.ylabel('distance from minimum')
    
    grad = defaultdict(lambda: [])
    newton = defaultdict(lambda: [])
    for b, points in all_results.items():
        for _, results in points.items():
            if results["gradient"]["x"] != inf and results["gradient"]["y"] != inf:
                grad[b].append(magnitude([1-results["gradient"]["x"], 1-results["gradient"]["y"]]))
            if results["newton"]["x"] != inf and results["newton"]["y"] != inf:
                newton[b].append(magnitude([1-results["newton"]["x"], 1-results["newton"]["y"]]))

    plt.scatter(list(grad.keys()), list(sum(i)/len(i) for i in grad.values()), label="Gradient")
    plt.scatter(list(newton.keys()), list(sum(i)/len(i) for i in newton.values()), label="Newton's method")
    plt.legend()
    plt.show()