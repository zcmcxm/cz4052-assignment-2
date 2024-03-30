import numpy as np

from pagerank_algos import pagerank, pagerank_closed_form
from utils import visualize_web_graph

if __name__=="__main__":
    # parameters
    damping_factor = 0.8  # teleporting probability
    transition_matrix = np.array([
        [0, 1/2, 0, 0],
        [1/3, 0, 0, 1/2],
        [1/3, 0, 1, 1/2],
        [1/3, 1/2, 0, 0]
    ]) # the four-webpages example from Slide 53 of Lecture 7

    num_of_webpages = transition_matrix.shape[1]
    E = np.array([[1 / num_of_webpages] * num_of_webpages]).reshape(num_of_webpages, -1) # distribution vector
    MAX_NUM_OF_ITER = 100
    TOLERENCE = 1e-6
    
    # init PageRank values
    init_pagerank_val = np.zeros((transition_matrix.shape[0], 1), dtype=float)
    for i in range(transition_matrix.shape[0]):
        init_pagerank_val[i] = float(1) / transition_matrix.shape[0]
    
    print(f'pagerank iterative result:\n {pagerank(transition_matrix, init_pagerank_val, damping_factor, E, MAX_NUM_OF_ITER, TOLERENCE)}')
    print(f'pagerank closed form result:\n {pagerank_closed_form(transition_matrix, damping_factor, E)}')

    visualize_web_graph(transition_matrix)