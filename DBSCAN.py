import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def dbscan(points, eps, min_points_in_cluster):
    neighbors_model = NearestNeighbors(radius=eps, algorithm='auto', leaf_size=30,
                                       metric='euclidean', metric_params=None, p=2,
                                       n_jobs=None)
    neighbors_model.fit(points)
    neighborhoods = neighbors_model.radius_neighbors(points, eps, return_distance=False)

    noise = -1
    unclassified = -2
    labels = np.zeros(points.shape[0], dtype=int) + unclassified
    curr_cluster_label = 0
    for i, label in enumerate(labels):
        if label == unclassified:
            seeds = list(neighborhoods[i])
            if len(seeds) < min_points_in_cluster:
                labels[i] = noise
            else:
                labels[seeds] = curr_cluster_label
                seeds.remove(i)
                while seeds:
                    curr_point_in_seeds = seeds.pop(0)
                    next_seeds = list(neighborhoods[curr_point_in_seeds])
                    if len(next_seeds) >= min_points_in_cluster:
                        for p in next_seeds:
                            if labels[p] in [unclassified, noise]:
                                if labels[p] == unclassified:
                                    seeds.append(p)
                                labels[p] = curr_cluster_label
                curr_cluster_label += 1
    return labels


if __name__ == "__main__":
    size = 100
    data = np.vstack([np.random.randn(size * 3 // 4, 2),
                      np.random.randn(size // 4, 2) / 2 + 2])

    labels = dbscan(data, 0.5, 4)
    print(np.unique(labels).shape[0])

    colors = ['k', 'b', 'r', 'g', 'c', 'm', 'y', 'silver', 'lime']
    for l, p in zip(labels, data):
        plt.plot(p[0], p[1], '.', color=colors[l + 1])
    plt.title('blobs data set visualisation')
    plt.show()
