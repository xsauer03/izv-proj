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

# TODO - uklid a priprava na odevzdani

def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    # create array
    data = np.arange(a, b, 1 / steps)
    return np.sum((data[1:] - data[:-1]) * f((data[:-1] + data[1:]) / 2))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    samples = 1000
    x = np.linspace(-3, 3, samples)
    y = np.array(a).reshape(-1, 1) ** 2 * x ** 3 * np.sin(x)

    plt.figure()

    for i in range(3):
        plt.fill_between(x, y[i], alpha=0.1)
        plt.annotate('$\int f_{' + a[i].__str__() + '}(X)dx = ' + round(np.trapz(y[i], x), 2).__str__() + '$',
                     xy=(x[-1], y[i][-1]))
        plt.plot(x, y[i], label=create_label(a[i]))

    plt.xlabel('$x$')
    plt.ylabel('$f_a(X)$')
    plt.xticks(np.arange(-3, 4, 1))
    plt.yticks(np.arange(0, 41, 5))
    plt.ylim(0)
    plt.xlim(min(x), 6)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=3)
    if (show_figure):
        plt.show()
    else:
        plt.savefig(save_path)


def create_label(i):
    return '$Y_{' + i.__str__() + '}(X)$'


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    samples = 1000
    t = np.linspace(0, 100, samples)
    f1 = 0.5 * np.cos((1 / 50) * np.pi * t)
    f2 = 0.25 * (np.sin(np.pi * t) + np.sin((3 / 2) * np.pi * t))
    f3 = f1 + f2

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(t, f1)
    ax1.set_yticks(np.arange(-0.8, 1.2, 0.4))
    ax1.set_xlim(0, 100)

    ax2.plot(t, f2)
    ax2.set_yticks(np.arange(-0.8, 1.2, 0.4))
    ax2.set_xlim(0, 100)

    ax3.plot(t, f3, color='green')
    ax3.set_yticks(np.arange(-0.8, 1.2, 0.4))
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_xlim(0, 100)

    mask = np.ma.masked_greater(f3, f1)
    ax3.plot(t, mask, color="red")

    if show_figure:
        plt.show()
    else:
        plt.savefig(save_path)


def download_data() -> List[Dict[str, Any]]:
    url = 'https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html'
    response = requests.get(url, allow_redirects=True)
    soup = BeautifulSoup(response.content, 'html.parser')

    table = []
    for row in soup.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.replace("°", "").replace(",", "."))
        table.append(row_data)

    table_dict = []
    for row in table:
        if len(row) >= 6:
            table_dict.append(dict(position=row[0], lat=float(row[2]), long=float(row[4]), height=float(row[6])))

    return table_dict


def main():
    def f(x): return 10 * x + 2

    # r = integrate(f, 0, 1, 100)
    # generate_graph([1., 1.5, 2.], show_figure=True,
    #                save_path="tmp_fn.png")
    # generate_sinus(show_figure=False, save_path="tmp_sin.png")
    download_data()
    print("Hello World!")


if __name__ == "__main__":
    main()
