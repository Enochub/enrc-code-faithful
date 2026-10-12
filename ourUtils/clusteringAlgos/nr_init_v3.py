import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster._kmeans import k_means
from sklearn.utils.extmath import row_norms
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import normalized_mutual_info_score as nmi
from .nr_kmeans import _determine_costs, _assign_labels, _update_centers_and_scatter_matrices, _update_rotation, NrKmeans


def init_strategy(data, clusterings, rounds, max_iter, random_state):
    nrkm = NrKmeans(clusterings, random_state=random_state,
                    max_iter=max_iter)
    nrkm.fit(data.detach().cpu().numpy(), rounds)
    return nrkm.centers, nrkm.P, nrkm.V
         

def init_kmeanspp_and_V_random(X, n_clusters, random_state):
    """
    Initialize the input parameters for a non-redundant kmeans like clustering algorithm.
    It returns a random rotation matrix and searches within these subspaces 
   
    :param X: input data
    :param n_clusters: list containing number of clusters for each subspace
    :param random_state: use a fixed random state to get a repeatable solution
    :return V: The orthogonal rotation matrix
    :return m: Dimensionality of each clustering
    :return P: The assinments of dimension (of the rotated space) to the clusterings
    :return centers the cluster centers for each clustering in the (unrotated) original space
    :return init_costs The cost of the clustering
    """
    data_dimensionality = X.shape[1]
    random_state = check_random_state(random_state)
    
    # Get number of subspaces
    subspaces = len(n_clusters)
    # Check if V is orthogonal

    V = ortho_group.rvs(dim=data_dimensionality,random_state=random_state)
    

    m = [int(data_dimensionality / subspaces)] * subspaces
    if data_dimensionality % subspaces != 0:
        choices = random_state.choice(range(subspaces), data_dimensionality - sum(m))
        for choice in choices:
            m[choice] += 1
            
    # Calculate projections P
    possible_projections = list(range(data_dimensionality))
    P = []
    for dimensionality in m:
        choices = random_state.choice(possible_projections, dimensionality, replace=False)
        P.append(choices)
        possible_projections = list(set(possible_projections) - set(choices))
        
    # Define initial cluster centers with kmeans++ for each subspace
    centers = []
    labels = [None] * subspaces
    for i in range(subspaces):
        k = n_clusters[i]
        if k > 1:
            P_subspace = P[i]
            cropped_X = np.matmul(X, V[:, P_subspace])
           
            centers_cropped = kmeans_plusplus(
                cropped_X,
                k,
                x_squared_norms=row_norms(cropped_X, squared=True),
                random_state=random_state,
            )[0]
            labels[i] =  pairwise_distances_argmin_min(X=cropped_X, Y=centers_cropped, metric='euclidean',\
                                                       metric_kwargs={'squared': True})[0]

            centers_sub = np.zeros((k, X.shape[1]))
            # Update cluster parameters
            for center_id, _ in enumerate(centers_sub):
                # Get points in this cluster
                points_in_cluster = np.where(labels[i] == center_id)[0]

                # Update center
                centers_sub[center_id] = np.average(X[points_in_cluster], axis=0)

            centers.append(centers_sub)
        else:
            centers.append(np.expand_dims(np.average(X, axis=0),0))
            labels[i] = np.zeros(X.shape[0], dtype=int)
    
    scatter_matrices = [None] * subspaces
    _update_all_scatter_matrices(X, subspaces, n_clusters, centers, labels, scatter_matrices)
    _perform_rotation(X, V, m,P , labels, subspaces, n_clusters, centers, scatter_matrices)

    
    for i in range(subspaces):
        k = n_clusters[i]
        if k > 1:
            P_subspace = P[i]
            cropped_X = np.matmul(X, V[:, P_subspace])
            labels[i] = k_means(cropped_X, k, init=np.matmul(centers[i], V[:, P_subspace]),n_init=1, random_state=random_state)[1]
            
            centers_sub = centers[i]
            # Update cluster parameters
            for center_id, _ in enumerate(centers_sub):
                points_in_cluster = np.where(labels[i] == center_id)[0]
                centers_sub[center_id] = np.average(X[points_in_cluster], axis=0)
        else:
            pass #We already have that one
        
    _update_all_scatter_matrices(X,subspaces, n_clusters, centers, labels, scatter_matrices)
    _perform_rotation(X,V,m,P,labels,subspaces, n_clusters, centers, scatter_matrices)
    
    # Assign each point to closest cluster center

    for i in range(subspaces):        
        labels[i] = _assign_labels(X, V, centers[i], P[i])
        # Update centers and scatter matrices depending on cluster assignments
        centers[i], scatter_matrices[i] = _update_centers_and_scatter_matrices(X, n_clusters[i], labels[i])

    init_costs = _determine_costs(scatter_matrices, P, V)

    return V, m, P, centers, init_costs




def _perform_rotation(X,V,m,P,labels, subspaces, n_clusters, centers, scatter_matrices):
    for i in range(subspaces - 1):
            for j in range(i + 1, subspaces):
                # Do rotation calculations
                P_1_new, P_2_new, V_new = _update_rotation(
                    X, V, i, j, n_clusters, labels, P, scatter_matrices, False)
                # Update V, m, P
                m[i] = len(P_1_new)
                m[j] = len(P_2_new)
                P[i] = P_1_new
                P[j] = P_2_new
                V = V_new
                
                
def _update_all_scatter_matrices(X, subspaces, n_clusters, centers, labels, scatter_matrices):
    for i in range(subspaces):
        # Calculate scatter matrices for the new V
        scatter_matrices[i] = _calculate_scatter_matrices(X,n_clusters[i], centers[i], labels[i]) 
    



def _calculate_scatter_matrices(X, n_clusters_subspace, centers, labels_subspace):
    scatter_matrices = np.zeros((n_clusters_subspace, X.shape[1], X.shape[1]))
    # Update cluster parameters
    for center_id, _ in enumerate(centers):
        # Get points in this cluster
        points_in_cluster = np.where(labels_subspace == center_id)[0]
        # Update scatter matrix
        centered_points = X[points_in_cluster] - centers[center_id]
        for entry in centered_points:
            scatter_matrices[center_id] += np.outer(entry, entry)
    return scatter_matrices
