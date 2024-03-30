import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def visualize_web_graph(M, node_size=1700):
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", edge_color="gray")
    plt.show()

def visualize_with_pyvis(M, output_filename="graph.html"):
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    net = Network(notebook=False, height="750px", width="100%")
    net.barnes_hut()
    net.from_nx(G)
    net.show(output_filename, notebook=False)