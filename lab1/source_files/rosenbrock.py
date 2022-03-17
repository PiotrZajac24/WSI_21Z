import matplotlib.pyplot as plt
import numpy as np
import random
from math import inf
import time
import argparse
from source_files.plots import plot_iterations, plot_time, plot_time_iterations, plot_accuracy, magnitude


# WSI LAB1 - Piotr Zajac


def timer(func):
    # decorator to measure time of function execution
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        stop = round(time.perf_counter() - start, 2)
        return result, stop
    return wrapper

def f(x, y):
    # Rosenbrock's function
    return (1-x)**2 + 100*(y - x**2)**2

def df_dx(x, y):
    # derivative of f(x, y) with respect to x
    return 2*(200*x**3 - 200*x*y + x - 1)

def df_dy(x, y):
    # derivative of f(x, y) with respect to y
    return 200*(y - x**2)

def df_dxx(x, y):
    # second derivative of f(x, y) with respect to x
    return 2*(600*x**2 - 200*y + 1)

def df_dxy(x, y):
    # second derivative of f(x, y) with respect to x and y
    return -400*x

def df_dyy(x, y):
    # second derivative of f(x, y) with respect to y
    return 200

@timer
def gradient_descent(x, y, max_iter, beta, epsilon):
    # gradient descent method
    gradient = [1, 1]   # initial step value (it will change before first iteration)
    iterations = 0
    
    try:
        while magnitude(gradient) > epsilon and iterations < max_iter:
            gradient = [beta*df_dx(x, y), beta*df_dy(x, y)] # gradient multiplied by beta
            x -= gradient[0]
            y -= gradient[1]
            iterations += 1
    except OverflowError:
        x, y = inf, inf
    return round(x, 2), round(y, 2), iterations

@timer
def newtons_method(x, y, max_iter, beta, epsilon):
    # Newton's method
    iterations = 0
    delta = [1, 1]  # initial step value (it will change before first iteration)
    
    try:
        while magnitude(delta) > epsilon and iterations < max_iter:
            gradient = [df_dx(x, y), df_dy(x, y)]
            hessian = [[df_dxx(x, y), df_dxy(x, y)], [df_dxy(x, y), df_dyy(x, y)]]
            delta = np.matmul(np.linalg.inv(hessian), gradient)
            delta = beta*delta  # step vector multiplied by beta
            x -= delta[0]
            y -= delta[1]
            iterations += 1
    except OverflowError:
        x, y = inf, inf
    return round(x, 2), round(y, 2), iterations


def plot():
    # show Rosenbrock's function in 3D
    plt.title("Rosenbrock's function")
    x_axis = np.linspace(-5, 5, 200)
    y_axis = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = f(X, Y)
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.plot_surface(X, Y, Z, cstride=1, cmap='jet', edgecolor='none')
    ax.plot(4, 4, 0)
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()

    default_beta = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25,
    0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.016, 0.014,
    0.012, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.0028, 0.0026, 0.0024, 0.0022, 0.002, 0.0018,
    0.0016, 0.0014, 0.0012, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]

    parser.add_argument('--points', default=30, type=int)
    parser.add_argument('--beta', default=default_beta, nargs='+', type=float)
    parser.add_argument('--iterations', default=30000, type=int)
    parser.add_argument('--epsilon', default=1e-12, type=float)
    args = parser.parse_args()

    if args.points <= 0:
        raise ValueError("There should be at least one point.")
    if any(not 0 < b < 1 for b in args.beta):
        raise ValueError("Beta should contain values between 0 and 1 (exclusive).")
    if args.iterations <= 0:
        raise ValueError("There should be positive number of iterations.")
    if args.epsilon <= 0:
        raise ValueError("Epsilon should contain positive value.")

    args.beta = sorted(list(set(args.beta)))[::-1]
    return args



def main():
    try:
        args = parse_arguments()
        random.seed(12323)  # comment this line to get different starting points than in report

        print(f"Max iterations: {args.iterations}")
        print(f"Beta: {args.beta}")
        print(f"Epsilon: {args.epsilon}")
        print()

        results = {b: {} for b in args.beta}

        for _ in range(args.points):
            x, y = random.uniform(-5, 5), random.uniform(-5, 5)
            for b in args.beta:
                print(f"Initial coordinates: ({round(x, 2)}, {round(y, 2)}), beta={b}")
                print("Gradient", end=' - ')

                results_grad, time_grad = gradient_descent(x, y, args.iterations, b, args.epsilon)
                x_grad, y_grad, iter_grad = results_grad    # sample result of gradient descent

                print(f"Final coordinates: ({x_grad}, {y_grad}), iterations: {iter_grad}, time: {time_grad}")
                print("Newton", end=' - ')

                results_newton, time_newton = newtons_method(x, y, args.iterations, b, args.epsilon)  # sample result of newton's method
                x_newton, y_newton, iter_newton = results_newton

                print(f"Final coordinates: ({x_newton}, {y_newton}), iterations: {iter_newton}, time: {time_newton}", )
                print(20*"_")

                results[b][f"({x}, {y})"] = {
                    "gradient": {
                        "x": x_grad, "y": y_grad, "distance": magnitude([x_grad-1, y_grad-1]) if x_grad != inf and y_grad != inf else inf,
                         "iterations": iter_grad, "time": time_grad},
                    "newton": {
                        "x": x_newton, "y": y_newton, "distance": magnitude([x_newton-1, y_newton-1]) if x_newton != inf and y_newton != inf else inf,
                        "iterations": iter_newton, "time": time_newton}
                }

        plot_iterations(results)
        plot_time(results)
        plot_time_iterations(results)
        plot_accuracy(results)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()