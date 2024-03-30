import numpy as np
import argparse
import matplotlib.pyplot as plt

from pagerank_algos import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=100, help="The size of the web graph (default: 500)")
    args = parser.parse_args()

    # parameters
    damping_factor_list = [round(x, 2) for x in list(np.arange(0.7, 0.96, 0.05))]
    size = args.size
    web_matrix = generate_random_web_matrix(size)
    num_of_webpages = web_matrix.shape[1]
    E = np.array([[1 / num_of_webpages] * num_of_webpages]).reshape(num_of_webpages, -1) # distribution vector
    MAX_NUM_OF_ITER = 100
    TOLERENCE = 1e-6
    colors_hex = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']
    
    # init PageRank values
    init_pagerank_val = np.zeros((web_matrix.shape[0], 1), dtype=float)
    for i in range(web_matrix.shape[0]):
        init_pagerank_val[i] = float(1) / web_matrix.shape[0]

    # convert web matrix to transition matrix
    transition_matrix = web_matrix / np.sum(web_matrix, axis=1)[:, np.newaxis]
    transition_matrix[np.isnan(transition_matrix)] = 0
    transition_matrix = transition_matrix.T

    data = {}

    for i in range(len(damping_factor_list)):
        data[i] = pagerank(transition_matrix, init_pagerank_val, damping_factor_list[i], E, MAX_NUM_OF_ITER, TOLERENCE)
    
    min_edge = min(min(data[i]) for i in range(len(damping_factor_list)))
    max_edge = max(max(data[i]) for i in range(len(damping_factor_list)))
    bins = np.logspace(np.log10(min_edge), np.log10(max_edge), 30)
    bins = bins.flatten()

    print(bins)

    for i in range(len(damping_factor_list)):
        hist, edges = np.histogram(data[i], bins=bins, density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:]) # Calculate bin centers
        width = np.diff(bins) # Normalize by bin width
        hist_norm = hist / width
        plt.plot(bin_centers, hist_norm, marker='o', linestyle='-', label=f'α = {damping_factor_list[i]}', color=colors_hex[i])

    plt.xlabel('PageRank Value')
    plt.ylabel('# of nodes (bin counts)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
