import numpy as np
from numpy import random as rand
from numpy import linalg as la


def init_farthest_models(points, cluster_number):
    models_index = [rand.randint(0, points.shape[0])]
    done = 1
    while done < cluster_number:
        far_models = 0
        max_distance_sum = 0.
        for i, p in enumerate(points):
            distance_sum = la.norm(points[models_index] - p, axis=1).sum()
            if distance_sum > max_distance_sum:
                max_distance_sum = distance_sum
                far_models = i
        models_index += [far_models]
        done += 1
    models_index.remove(models_index[0])
    far_models = 0
    max_distance_sum = 0.
    for i, p in enumerate(points):
        distance_sum = la.norm(points[models_index] - p, axis=1).sum()
        if distance_sum > max_distance_sum:
            max_distance_sum = distance_sum
            far_models = i
    models_index += [far_models]
    return models_index


def init_nearest_models(points, cluster_number):
    models_index = [rand.randint(0, points.shape[0])]
    done = 1
    while done < cluster_number:
        near_models = 0
        min_max_distance_sum = 100.
        for i, p in enumerate(points):
            if i in models_index:
                continue
            distance_sum = la.norm(points[models_index] - p, axis=1).sum()
            if distance_sum < min_max_distance_sum:
                min_max_distance_sum = distance_sum
                near_models = i
        models_index += [near_models]
        done += 1
    return models_index


def kmeans(points, cluster_number, do_radius_norm=True, init_type="rand"):
    if init_type == "rand":
        initial_models_index = rand.randint(0, points.shape[0], cluster_number)
    elif init_type == "near":
        initial_models_index = init_nearest_models(points, cluster_number)
    else:
        initial_models_index = init_farthest_models(points, cluster_number)
    models = points[initial_models_index]
    clusters_radius = np.ones(cluster_number, dtype=np.float32)
    labels = np.empty(points.shape[0], dtype=int)
    for iteration in range(100):
        print(iteration)
        last_models = models
        models = np.zeros([cluster_number, points.shape[1]], dtype=np.float32)
        clusters_size = np.zeros(cluster_number, dtype=int)
        for i, p in enumerate(points):
            n = np.argmin(la.norm(last_models - p, axis=1) / clusters_radius)
            labels[i] = n
            clusters_size[n] += 1
            models[n] += p
        empty_clusters = clusters_size == 0
        models[empty_clusters] = last_models[empty_clusters]
        clusters_size[empty_clusters] = 1
        models /= clusters_size[:, None]
        if (last_models == models).all():
            break
        if do_radius_norm:
            distance = la.norm(points - models[labels], axis=1)
            clusters_radius = np.zeros([cluster_number])
            for i in range(cluster_number):
                clusters_radius[i] = np.sum(distance[labels == i])
            clusters_radius = np.sqrt(clusters_radius / clusters_size)
            clusters_radius[clusters_radius == 0] = 1
    return labels, models


def purity(cluster_number_min, cluster_number_max, points, true_labels, do_radius_norm=True):
    metrics = np.zeros(cluster_number_max - cluster_number_min + 1)
    for i, cluster_number in enumerate(range(cluster_number_min, cluster_number_max + 1)):
        labels, _ = kmeans(points, cluster_number, do_radius_norm)
        for j in range(cluster_number):
            true_cluster_labels = true_labels[labels == j]
            ones = np.sum(true_cluster_labels)
            zeros = np.size(true_cluster_labels) - ones
            metrics[i] += max(ones, zeros)
    return metrics / points.shape[0]
