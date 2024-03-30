import numpy as np

def pagerank(transition_matrix, init_pagerank_val, damping_factor, distribution_matrix, max_iter, tolerance=1e-6):
    prev_pagerank_val = init_pagerank_val.copy()

    for i in range(max_iter):
        pagerank_val = damping_factor * np.dot(transition_matrix, prev_pagerank_val) + (1 - damping_factor) * distribution_matrix

        difference = np.linalg.norm(pagerank_val - prev_pagerank_val)

        # Check for convergence
        if difference < tolerance:
            print(f"Convergence reached after {i+1} iterations.")
            return pagerank_val
        prev_pagerank_val = pagerank_val.copy()
    print("Maximum number of iterations reached without convergence.")
    return pagerank_val

def pagerank_closed_form(transition_matrix, damping_factor, distribution_matrix):
    num_pages = transition_matrix.shape[0]
    I = np.diag(np.full(num_pages,1))
    b = (1 - damping_factor) * distribution_matrix
    return np.dot(np.linalg.inv(I - damping_factor * transition_matrix), b)  

def generate_random_web_matrix(size, p=0.5 ,seed=9):
    np.random.seed(seed)
    # randomly generate a number between 0 and 1 and compare it with p to decide whether there's a link
    return np.random.rand(size,size) < p