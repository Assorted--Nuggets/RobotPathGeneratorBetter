import numpy as np
import matplotlib.pyplot as plt


def line(p1, p2):
    return lambda t: p1(t) + (p2(t) - p1(t))*t


def bezier_func(points):
    p = lambda k: lambda t: points[k]
    lines = [p(k) for k in range(0, len(points))]
    # We compose lines n - 1 times
    for i in range(0, len(points)-1):
        # What we basically do here is "squash" the lines to make new lines
        for j in range(0, len(lines)-1):
            lines[j] = line(lines[j], lines[j+1])
        # remove the last index
        lines = lines[:-1]
    return lines[0]


def build_bezier(points, resolution=1000):
    curve = bezier_func(points)
    domain = np.linspace(0, 1, num=resolution)
    return np.apply_along_axis(curve, 0, domain.reshape((1, resolution))), curve


def plot_bezier():
    points = np.array([[0, 0],
                       [0.2, 1],
                       [1, 0.5],
                       [0.5, 0.25],
                       [0.12, 0.5],
                       [0.6, 0.8]])

    curve_points = build_bezier(points)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(curve_points[0], curve_points[1])
    plt.show()

