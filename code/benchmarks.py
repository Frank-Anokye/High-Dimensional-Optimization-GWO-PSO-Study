import numpy as np


def spherical_function(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def shifted_spherical_function(x: np.ndarray) -> float:
    shift = np.sqrt(2)
    return float(np.sum((x - shift) ** 2))


def rastrigin_function(x: np.ndarray) -> float:
    d = len(x)
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock_function(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


BENCHMARKS = {
    "Sphere": {
        "func": spherical_function,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
        "class": "Unimodal, convex, separable",
    },
    "Shifted Sphere": {
        "func": shifted_spherical_function,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
        "class": "Unimodal, convex, separable, shifted optimum",
    },
    "Rastrigin": {
        "func": rastrigin_function,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
        "class": "Multimodal, non-convex, separable",
    },
    "Rosenbrock": {
        "func": rosenbrock_function,
        "bounds": (-2.048, 2.048),
        "optimum": 0.0,
        "class": "Unimodal, non-convex, non-separable, narrow curved valley",
    },
}
