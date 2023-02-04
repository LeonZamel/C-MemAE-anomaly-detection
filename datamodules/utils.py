import copy

import numpy as np
import torch
from sklearn.utils import shuffle


def process_for_anomalydetection(
    dataset_train, dataset_test, targets, num_train_samples, test_proportion_anomaly
):
    """
    Fix the target and set the number of training samples for training data.
    Adjust the anomaly proportion for testing data.
    """

    # Get the indices of normal samples
    train_normals_idx = (dataset_train.targets == torch.tensor(targets).unsqueeze(1)).any(0)
    # Only use normal samples for training
    dataset_train.data = dataset_train.data[train_normals_idx]
    dataset_train.targets = dataset_train.targets[train_normals_idx]

    # If self.num_train_samples has a strictly positive value, we only use that number of training samples
    # This is used e.g. for few-shot learning
    if num_train_samples > 0:
        dataset_train.data, dataset_train.targets = shuffle(dataset_train.data, dataset_train.targets)
        dataset_train.data = dataset_train.data[:num_train_samples]
        dataset_train.targets = dataset_train.targets[:num_train_samples]

    datasets_test = []
    for target in targets:
        ds_test = copy.deepcopy(dataset_test)
        test_normals_idx = ds_test.targets == torch.tensor(target)
        datasets_test.append(fix_anomaly_proportion(ds_test, test_normals_idx, test_proportion_anomaly))

    return dataset_train, datasets_test


def fix_anomaly_proportion(dataset, normals, proportion_anomaly):
    """
    Take a data set and a vector of 0 and 1 values indicating which samples are normal (indicated by a 1)
    make sure that the proportion of anomalies is ~= to the proportion given
    """
    normals_X = dataset.data[normals]
    normals_y = dataset.targets[normals]

    anomalies_X = dataset.data[~normals]
    anomalies_y = dataset.targets[~normals]

    # Determine how many anomalous samples are needed such that
    # num_anomalies / len(dataset) = proportion_anomalies
    # num_anomalies / (num_normals + num_anomalies) = proportion_anomalies
    # num_anomalies = proportion_anomalies * (num_normals + num_anomalies)
    # num_anomalies = proportion_anomalies * num_normals + proportion_anomalies * num_anomalies
    # (1-proportion_anomalies) * num_anomalies = proportion_anomalies * num_normals
    # num_anomalies = proportion_anomalies * num_normals / (1-proportion_anomalies)
    num_anomalies = int(len(normals_X) * proportion_anomaly / (1 - proportion_anomaly))

    # Shuffle and pick anomalies
    anomalies_X, anomalies_y = shuffle(anomalies_X, anomalies_y)
    anomalies_X = anomalies_X[:num_anomalies]
    anomalies_y = anomalies_y[:num_anomalies]

    # Put normal and anomalous samples back together, shuffle again
    if isinstance(normals_X, np.ndarray):
        X = np.concatenate((normals_X, anomalies_X), axis=0)
    else:
        X = torch.cat((normals_X, anomalies_X), dim=0)
    y = torch.cat((normals_y, anomalies_y), dim=0)
    X, y = shuffle(X, y)

    dataset.data = X
    dataset.targets = y

    return dataset
