########## Machine Learning In Web - Exercise 3, Practical(b) ##########
## Author : Hakki Egemen Gülpinar
## Date: 28.11.2024
## Subject: Clustering
#########################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel


def create_circular_data(n_samples=300):
    
    X, y = make_blobs(n_samples=n_samples, #randomly generate data points to distribute in clusters
                      centers=10,
                      cluster_std=0.4,
                      random_state=31)
    return X, y

def create_checkerboard(size=30, squares=4): #creates a checkerboard pattern, it generates more complex data.
                                             #the reason is: test Spectral Clustering method with non-linear data
    x = np.linspace(0, squares, size)
    y = np.linspace(0, squares, size)
    xx, yy = np.meshgrid(x, y)
    X = np.column_stack([xx.ravel(), yy.ravel()])
    y = np.mod(np.floor(xx.ravel()) + np.floor(yy.ravel()), 2)
    return X, y


X1, y1 = create_circular_data()
X2, y2 = create_checkerboard()


scaler = StandardScaler() #normalization of the features of data. This is required for clustering algorithms.(K-Means, Affinity Propagation, Spectral Clustering)
X1_scaled = scaler.fit_transform(X1)
X2_scaled = scaler.fit_transform(X2)


similarity1 = rbf_kernel(X1_scaled, gamma=1.0)#kernel(X1_scaled, gamma=1.0) #creates similarity matrix with radial basis function kernel for Spectral Clustering method
similarity2 = rbf_kernel(X2_scaled, gamma=1.0)#kernel(X2_scaled, gamma=1.0) #rbf assign a score to each pair of data points, which is used to determine the similarity between them.
                                               #gamma : sensitive parameter for similarity calculation. 

n_clusters = 10
damping = 0.9 #it helps to prevent numerical oscillations, algorithm divergence(higher changes) in the similarity matrix. It is used in Affinity Propagation method.
#lower damping values are faster to converge but sensitive to noise.
#higer damping values are more stable but slower to converge.


def apply_clustering(X, similarity_matrix):
    # K-means++
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42) #in brief, k-means++ specify the initial cluster centers and then assign each point to the nearest cluster
    kmeans_labels = kmeans.fit_predict(X)
    
    # Affinity Propagation
    af = AffinityPropagation(damping=damping, random_state=42) #in brief, depending on the similarity of the data, it assign cluster centers automatically.
    af_labels = af.fit_predict(X)
    
    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_clusters,  #in brief, it uses the eigenvalues of a similarity matrix to reduce the dimensionality of the data before clustering in a lower dimensional space.
                                 affinity='precomputed',  #especially for recognise non-lineer clusterings
                                 random_state=42)
    spectral_labels = spectral.fit_predict(similarity_matrix)
    
    return kmeans_labels, af_labels, spectral_labels


def plot_results(X, labels_list, titles, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Clustering Results for {dataset_name}')
    
    for ax, labels, title in zip(axes, labels_list, titles):
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20')
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()


def evaluate_clustering(X, labels):
    if len(np.unique(labels)) < 2:
        return None, None
    
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    return silhouette, davies_bouldin


results_data = []

for X, similarity, name in [(X1_scaled, similarity1, 'Circular Dataset'),
                          (X2_scaled, similarity2, 'Checkerboard Dataset')]:
    
    kmeans_labels, af_labels, spectral_labels = apply_clustering(X, similarity)
    labels_list = [kmeans_labels, af_labels, spectral_labels]
    methods = ['K-means++', 'Affinity Propagation', 'Spectral Clustering']

    
    plot_results(X, labels_list, methods, name)
    
    
    for method, labels in zip(methods, labels_list):
        scores = evaluate_clustering(X, labels)
        if scores:
            results_data.append({
                'Dataset': name,
                'Method': method,
                'Silhouette Score': f"{scores[0]:.3f}", # calculates quality of clustering, higher is better
                'Davies-Bouldin Score': f"{scores[1]:.3f}" # calculates distance between clusters, lower is better
            })


results_df = pd.DataFrame(results_data)
plt.figure(figsize=(10, 6))


plt.subplot(211)
plt.axis('off')
plt.title("Clustering Algorithm Comparison Results", pad=0, fontsize=14, fontweight='bold')

plt.subplot(212)
ax = plt.gca()
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=results_df.values,
                colLabels=results_df.columns,
                cellLoc='center',
                loc='center')


table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))

for cell in table._cells.values():
    cell.set_height(0.2)

plt.subplots_adjust(top=0.90, bottom=0.45)
plt.show()

#overral table results of clustering algorithms
#K-means++ is the best algorithm for Checkerboard Dataset,  but overall general results are not good for this dataset because of complex data.
#
#  Affinity Propagation is the best algorithm for Circular Dataset, 
# that means clusters are not well separated in Silhouuette Score, David-Bouldin Score means frequency of data points are more in clusters.