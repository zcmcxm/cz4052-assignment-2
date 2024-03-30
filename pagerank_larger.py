import numpy as np
import argparse
import matplotlib.pyplot as plt

from pagerank_algos import *
from utils import visualize_with_pyvis

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=100, help="The size of the web graph (default: 500)")
    args = parser.parse_args()

    # parameters
    damping_factor = 0.8  # teleporting probability
    size = args.size
    web_matrix = generate_random_web_matrix(size)
    num_of_webpages = web_matrix.shape[1]
    E = np.array([[1 / num_of_webpages] * num_of_webpages]).reshape(num_of_webpages, -1) # distribution vector
    MAX_NUM_OF_ITER = 100
    TOLERENCE = 1e-6
    
    # init PageRank values
    init_pagerank_val = np.zeros((web_matrix.shape[0], 1), dtype=float)
    for i in range(web_matrix.shape[0]):
        init_pagerank_val[i] = float(1) / web_matrix.shape[0]

    # convert web matrix to transition matrix
    transition_matrix = web_matrix / np.sum(web_matrix, axis=1)[:, np.newaxis]
    transition_matrix[np.isnan(transition_matrix)] = 0
    transition_matrix = transition_matrix.T

    # pagerank_scores, total_difference = pagerank_with_difference(transition_matrix, init_pagerank_val, damping_factor, E, MAX_NUM_OF_ITER, TOLERENCE)
    pagerank_scores = pagerank(transition_matrix, init_pagerank_val, damping_factor, E, MAX_NUM_OF_ITER, TOLERENCE)

    print(f'pagerank iterative result:\n {pagerank_scores}')

    # visualize_with_pyvis(web_matrix)

    plt.hist(pagerank_scores, bins=20, edgecolor='black')
    plt.xlabel('PageRank Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final PageRank Values')
    plt.grid(True)
    plt.show()