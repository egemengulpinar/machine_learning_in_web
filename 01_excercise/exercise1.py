########## Machine Learning In Web - Exercise 1, Practical(b) ##########
## Author : Hakki Egemen Gülpinar, Szymon Czajkowskis
## Date: 27.10.2024
## Subject: Demonstration different characteristics of the distribution of the page rank according to Erdos/Renyi and Barabasi/Albert graph models
#########################################################



import networkx as nx
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import poisson

def draw_graph(
    pagerank_er: nx.Graph,
    pagerank_ba: nx.Graph,     
    diameter_manual_BA: float,
    diameter_manual_ER: float,
    coefficent_manual_BA: float,
    coefficent_manual_ER: float,
    pos_er: Dict[int, Tuple[float, float]], 
    pos_ba: Dict[int, Tuple[float, float]],
    poisson_list: List) -> None:

    # Set up the figure for visualizing graphs
    plt.figure(figsize=(30, 15))

    # Erdős-Rényi Graph - Node sizes and labels based on PageRank values
    plt.subplot(1, 2, 1)
    nx.draw_networkx_nodes(ER_graph, pos_er, node_size=[v * 3000 for v in pagerank_er.values()], 
                           node_color='skyblue', label="Nodes (scaled by PageRank)")
    nx.draw_networkx_labels(ER_graph, pos_er, labels={node: f"{pagerank_er[node]:.2f}" for node in ER_graph.nodes()}, 
                            font_size=8)
    nx.draw_networkx_edges(ER_graph, pos_er, alpha=0.5, edge_color='purple', label="Edges")
    plt.title("Erdős-Rényi Graph (Exponential)")
    plt.legend(scatterpoints=1, loc="upper right", fontsize=10)

    # Barabási-Albert Graph - Node sizes and labels based on PageRank values
    plt.subplot(1, 2, 2)
    nx.draw_networkx_nodes(BA_graph, pos_ba, node_size=[v * 3000 for v in pagerank_ba.values()], 
                           node_color='gold', label="Nodes (scaled by PageRank)")
    nx.draw_networkx_labels(BA_graph, pos_ba, labels={node: f"{pagerank_ba[node]:.2f}" for node in BA_graph.nodes()}, 
                            font_size=8)
    nx.draw_networkx_edges(BA_graph, pos_ba, alpha=0.5, edge_color='purple', label="Edges")
    plt.title("Barabási-Albert Graph")
    plt.legend(scatterpoints=1, loc="upper right", fontsize=10)

    # Calculate mean and variance for PageRank values
    pr_values_er = list(pagerank_er.values())
    pr_values_ba = list(pagerank_ba.values())
    mean_er = np.mean(pr_values_er)
    var_er = np.var(pr_values_er)
    mean_ba = np.mean(pr_values_ba)
    var_ba = np.var(pr_values_ba)

    print("Erdős-Rényi PageRank ---> Mean:", mean_er, "  Variance:", var_er)
    print("Barabási-Albert PageRank ---> Mean:", mean_ba, "  Variance:", var_ba)

    # Plot histograms for PageRank distributions
    plt.figure(figsize=(30, 15))

    plt.subplot(1, 2, 1)
    plt.hist(pr_values_er, bins=30, color='skyblue', edgecolor='black')
    plt.title("Erdős-Rényi PageRank Distribution")
    plt.xlabel("PageRank Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(pr_values_ba, bins=30, color='orange', edgecolor='black')
    plt.title("Barabási-Albert PageRank Distribution")
    plt.xlabel("PageRank Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.bar(range(len(poisson_list)), poisson_list, color='skyblue', alpha=0.9)
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title('Poisson Distribution for Degrees in Erdős-Rényi Model')
    plt.show()

    # Additional plot for comparing means and variances of PageRank distributions
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))

    # Plot Erdős-Rényi Variance and Barabási-Albert Variance
    ax1.bar(["Erdős-Rényi Variance"], [var_er], color='skyblue', label="Erdős-Rényi Variance", width=0.1)
    ax1.bar(["Barabási-Albert Variance"], [var_ba], color='orange', label="Barabási-Albert Variance", width=0.1)
    ax1.set_title("Comparison of Variance for PageRank Distributions")
    ax1.set_ylabel("Value")
    ax1.legend()

    # Plot Erdős-Rényi Mean and Barabási-Albert Mean  
    ax2.bar(["Erdős-Rényi Mean"], [mean_er], color='skyblue', label="Erdős-Rényi Mean", width=0.1)
    ax2.bar(["Barabási-Albert Mean"], [mean_ba], color='orange', label="Barabási-Albert Mean", width=0.1)
    ax2.set_title("Comparison of Mean for PageRank Distributions")
    ax2.set_ylabel("Value")
    ax2.legend()

    # Plot Erdős-Rényi Mean and Barabási-Albert Diameter 
    ax3.bar(["Erdős-Rényi Diameter"], [diameter_manual_ER], color='skyblue', label="Erdős-Rényi Diameter", width=0.1)
    ax3.bar(["Barabási-Albert Diameter"], [diameter_manual_BA], color='orange', label="Barabási-Albert Diameter", width=0.1)
    ax3.set_title("Comparison of Diameters")
    ax3.set_ylabel("Value")
    ax3.legend()


    # Plot Erdős-Rényi Mean and Barabási-Albert Cluster coefficient
    ax4.bar(["Erdős-Rényi Coefficient"], [coefficent_manual_ER], color='skyblue', label="Erdős-Rényi Coefficient", width=0.1)
    ax4.bar(["Barabási-Albert Coefficient"], [coefficent_manual_BA], color='orange', label="Barabási-Albert Coefficient", width=0.1)
    ax4.set_title("Comparison of Coefficients")
    ax4.set_ylabel("Value")
    ax4.legend()
    plt.tight_layout()
    plt.show()
    return 0

def calculate_pagerank(
    ER_graph: nx.Graph, 
    BA_graph: nx.Graph, 
) -> Dict:
    # Calculate PageRank for the Erdős-Rényi graph
    pagerank_er = nx.pagerank(ER_graph)
    pagerank_ba = nx.pagerank(BA_graph)

    return pagerank_er, pagerank_ba

    

# Parameters
num_nodes = 100  # Number of nodes
avg_degree = 10  # Average degree

# Create Erdős-Rényi graph
# p = probablity of the edge between two nodes
p = avg_degree / (num_nodes - 1) #The equation for the expected value in a binomial distribution is E(X) = np, so N-1 is the possible number of the nodes except for one node itself.
ER_graph = nx.erdos_renyi_graph(num_nodes, p)
pos_er = nx.spring_layout(ER_graph)
 
diameter_manual_ER = math.log2(num_nodes) # The diameter of the Erdős-Rényi graph is approximately log2(N)
coefficent_manual_ER = 1 / num_nodes # The clustering coefficient of the Erdős-Rényi graph is approximately 1 / N
print("manual_diameter: {}".format(diameter_manual_ER))
print("manual_coefficient: {}".format(coefficent_manual_ER))
print("Total possible edges : ", num_nodes*(num_nodes-1)/2) # Total possible edges in the graph, N*(N-1))/2

# Create Barabási-Albert graph with networkx library
m = avg_degree // 2 # The reason of dividing by 2 is that the Barabási-Albert model, each edge contributes to the degree of two nodes.
BA_graph = nx.barabasi_albert_graph(num_nodes, m)
diameter_manual_BA = math.log2(num_nodes) / math.log2(math.log2(num_nodes)) # The diameter of the Barabási-Albert graph is approximately log2(N) / log2(log2(N))    
coefficent_manual_BA = 1 / (num_nodes**0.75) # The clustering coefficient of the Barabási-Albert graph is approximately 1 / N^0.75
print("manual_diameter: {}".format(diameter_manual_BA))
print("manual_coefficient: {}".format(coefficent_manual_BA))
# Calculate the position of nodes in the graph
pos_ba = nx.spring_layout(BA_graph)

# Calculate PageRank
pagerank_values = calculate_pagerank(ER_graph=ER_graph, BA_graph= BA_graph)
# (OPTIONAL) Calculate and Analyze Poission Distribution for Erdos-Renyi Model
degree_sequence = [d for n, d in ER_graph.degree()]
poisson_dist = poisson(avg_degree)
poisson_list = [poisson_dist.pmf(k) for k in range(max(degree_sequence) + 1)] # Calculate the Poisson distribution for the degrees in the Erdos-Renyi model. It helps us to explain more clearly the random distribution of node degrees in the Erdős-Rényi graph.


# Visaualize the graphs and PageRank values
draw_graph(pagerank_er= pagerank_values[0], pagerank_ba= pagerank_values[1], 
           diameter_manual_BA= diameter_manual_BA,
           diameter_manual_ER= diameter_manual_ER,
           coefficent_manual_BA = coefficent_manual_BA,
           coefficent_manual_ER = coefficent_manual_ER, 
           pos_er= pos_er, pos_ba= pos_ba,
           poisson_list = poisson_list)
