import numpy as np
from scipy.optimize import minimize
import random


# Funkcja celu
def f(x):
    x1, x2 = x
    return (x1 - 1) * (x2 - 1) * x1 * x2


# Funkcja ograniczenia
def g(x):
    x1, x2 = x
    return (x1 - 1) ** 2 + x2 ** 2 - 1


# Funkcja kary, suma kwadratów ograniczeń
def S(x):
    return max(0, g(x)) ** 2


# Gradient funkcji celu
def gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = np.array(x)
        x2 = np.array(x)
        x1[i] += h
        x2[i] -= h
        grad[i] = (f(x1) - f(x2)) / (2 * h)
    return grad


# Metoda gradientu z adaptacyjnym krokiem
def gradient_descent_adaptive_step(f, initial_point, tolerance, initial_learning_rate=1):
    max_iterations = 1000
    point = np.array(initial_point)
    for i in range(max_iterations):
        learning_rate = initial_learning_rate
        grad = gradient(f, point)

        new_point = point - learning_rate * grad
        zabezpieczenie = 0
        while f(new_point) >= f(point):
            learning_rate /= 2
            new_point = point - learning_rate * grad
            zabezpieczenie += 1
            if zabezpieczenie > 1000:
                break

        print(
            f"\tIteration {i + 1}: Point = {point}, New Point = {new_point}, Function value = {f(point)}, Step = {learning_rate}")

        if np.linalg.norm(new_point - point) < tolerance:
            point = new_point
            break

        point = new_point

    return point


# Parametry
initial_point = [0.5, 0.5]
tolerance = 1e-3
adaptive_learning_rate = 1
initial_c = 1

# Generowanie losowych punktów startowych
points = []
for i in range(10):
    random_x = [random.uniform(-10, 10), random.uniform(-10, 10)]
    points.append(random_x)


# Funkcja kary zewnętrznej
def kara_zewnetrzna(start_point):
    max_iterations = 1000
    point = np.array(start_point)
    c = initial_c
    for i in range(max_iterations):

        F = lambda x: f(x) + c * S(x)
        new_point = gradient_descent_adaptive_step(F, point, tolerance, adaptive_learning_rate)

        print(f"Iteration {i + 1}: Point = {point}, New Point = {new_point}, Function value = {f(point)}, c = {c}")

        if np.linalg.norm(new_point - point) < tolerance:
            point = new_point
            break

        point = new_point
        c *= 2

    return point


print("\nKARA ZEWNETRZNA:")
minimum_point = kara_zewnetrzna(initial_point)
print(f"Minimum znalezione w punkcie: {minimum_point}")
print(f"Wartość funkcji w minimum: {f(minimum_point)}")

print("\nKARA ZEWNETRZNA WIECEJ PUNKTOW:")
min_point = points[0]
min_start_point = points[0]
for start_point in points:
    minimum_point = kara_zewnetrzna(start_point)
    if f(minimum_point) < f(min_point):
        min_point = minimum_point
        min_start_point = start_point
print(f"Wystartowano z punktu: {min_start_point}")
print(f"Minimum znalezione w punkcie: {min_point}")
print(f"Wartość funkcji w minimum: {f(min_point)}")
