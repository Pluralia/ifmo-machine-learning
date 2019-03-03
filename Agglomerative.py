import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


class AgglomerativeClustering:
    def __init__(self, points):
        self.point_number = points.shape[0]
        self.points = points
        self.pick_up_collections_of_clusters(self.count_distanse())

    def count_distanse(self):
        distance_matrix = euclidean_distances(self.points, self.points)
        sorted_all_distance = np.vstack(np.unravel_index(np.argsort(distance_matrix.ravel()),
                                                         (self.point_number, self.point_number))).T
        sorted_all_distance = sorted_all_distance[sorted_all_distance[:, 0] - sorted_all_distance[:, 1] < 0]
        distance_vector = np.array([distance_matrix[tuple(i)]
                                    for i in sorted_all_distance]).reshape(sorted_all_distance.shape[0], 1)
        return np.hstack([sorted_all_distance, distance_vector])

    def pick_up_collections_of_clusters(self, sorted_distance):
        current_collection_of_clusters = [[i] for i in range(self.point_number)]
        self.collections_of_clusters = [current_collection_of_clusters.copy()]
        while True:
            nearest_points = sorted_distance[0]
            cluster_index0 = self.get_cluster(current_collection_of_clusters, nearest_points[0])
            cluster_index1 = self.get_cluster(current_collection_of_clusters, nearest_points[1])
            if cluster_index0 != cluster_index1:
                print(int(100 * len(self.collections_of_clusters) / self.point_number), '%')
                if cluster_index0 < cluster_index1:
                    cluster_index0, cluster_index1 = cluster_index1, cluster_index0
                cluster0 = current_collection_of_clusters.pop(cluster_index0)
                cluster1 = current_collection_of_clusters.pop(cluster_index1)
                new_cluster = cluster0 + cluster1
                current_collection_of_clusters.append(new_cluster)
                self.collections_of_clusters.append(current_collection_of_clusters.copy())
                if len(current_collection_of_clusters) == 1:
                    break
                sorted_distance = self.update_distance(sorted_distance, new_cluster, current_collection_of_clusters)
        print('100 %')

    @staticmethod
    def get_cluster(collection_of_clusters, point_index):
        for i, cluster in enumerate(collection_of_clusters):
            if point_index in cluster:
                return i

    def update_distance(self, sorted_distance, new_cluster, collection_of_clusters):
        needed_index = np.ones(sorted_distance.shape[0], dtype='bool')
        new_cluster_center = np.mean(self.points[new_cluster], axis=0).reshape(1, 2)
        for i, points_dist in enumerate(sorted_distance):
            if points_dist[0] in new_cluster and points_dist[1] in new_cluster:
                needed_index[i] = False
                continue
            if points_dist[0] in new_cluster:
                cluster = collection_of_clusters[self.get_cluster(collection_of_clusters, points_dist[1])]
                cluster_center = np.mean(self.points[cluster], axis=0).reshape(1, 2)
                sorted_distance[i, 2] = np.linalg.norm(cluster_center - new_cluster_center)
            if points_dist[1] in new_cluster:
                cluster = collection_of_clusters[self.get_cluster(collection_of_clusters, points_dist[0])]
                cluster_center = np.mean(self.points[cluster], axis=0).reshape(1, 2)
                sorted_distance[i, 2] = np.linalg.norm(cluster_center - new_cluster_center)
        sorted_distance = sorted_distance[needed_index]
        sorted_distance = sorted_distance[np.argsort(sorted_distance[:, 2])]
        return sorted_distance

    def get_labels(self, cluster_number):
        collection_of_clusters = self.collections_of_clusters[-cluster_number]
        labels = np.empty(self.point_number, dtype=int)
        for i, cluster in enumerate(collection_of_clusters):
            for point_index in cluster:
                labels[point_index] = i
        pass
        return labels


def main():
    size = 100
    data = np.vstack([np.random.randn(size * 3 // 4, 2),
                      np.random.randn(size // 4, 2) / 2 + 2])

    clustering = AgglomerativeClustering(data)
    labels = clustering.get_labels(4)

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'silver', 'lime']
    for l, p in zip(labels, data):
        plt.plot(p[0], p[1], '.', color=colors[l])
    plt.title('blobs data set visualisation')
    plt.show()


if __name__ == "__main__":
    main()
