#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xsauer03

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    # create array
    data = np.arange(a, b, 1 / steps)
    return np.sum((data[1:] - data[:-1]) * f((data[:-1] + data[1:]) / 2))
    # return summary2


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    # print(np.array(np.linspace(-3,3),np.linspace(-3,3),np.linspace(-3,3)))
    # print(np.array(np.linspace(-3,3), np.linspace(-3,3),np.linspace(-3,3)))
    samples = 1000
    x = np.linspace(-3, 3, samples)
    y = np.array(a).reshape(-1, 1) ** 2 * x ** 3 * np.sin(x)
    # print(y[0])

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica"
    # })

    plt.figure()

    for i in range(3):
        plt.plot(x, y[i], label=create_label(a[i]))
        plt.fill_between(x, y[i], alpha=0.1)

    plt.ylim(0)
    plt.xlim(min(x))
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=3)
    plt.show()
    print(y)


def create_label(i):
    return '$Y_{' + i.__str__() + '}(X)$'


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    pass


def download_data() -> List[Dict[str, Any]]:
    pass


def main():
    def f(x): return 10 * x + 2

    # r = integrate(f, 0, 1, 100)
    generate_graph([1., 1.5, 2.], show_figure=False,
                   save_path="tmp_fn.png")
    print("Hello World!")


if __name__ == "__main__":
    main()
