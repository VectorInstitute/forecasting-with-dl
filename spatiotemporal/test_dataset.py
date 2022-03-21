import pytest
import torch
import numpy as np

from dataset_impls.weatherbench import WeatherBench

"""
Testing the dataset(s) is quite difficult, and it's recommended to visualize the data going into the network and coming out of it.
Note that all tests here are very hard-coded for the specific dataset use case.
"""


def test_weatherbench():
    train_args = [
        "tcc", 
        "/ssd003/projects/aieng/datasets/forecasting",
        "train",
        2,
        12,
        12,
        8
    ]
    val_args = [
        "tcc", 
        "/ssd003/projects/aieng/datasets/forecasting",
        "val",
        2,
        12,
        12,
        8
    ]
    train_dset = WeatherBench(*train_args)
    val_dset = WeatherBench(*val_args)

    # Check that splits are mutex
    train_data_idxs = np.array(train_dset.data_idxs)
    val_data_idxs = np.array(val_dset.data_idxs)
    assert np.all(
        np.intersect1d(train_data_idxs.flatten(), val_data_idxs.flatten(), assume_unique=True)
    )

    # Indexing takes a long time; put a bunch of examples in RAM
    train_idxs = np.array([train_dset._get_idxs(i) for i in range(240)])
    val_idxs = np.array([val_dset._get_idxs(i) for i in range(240)])
    assert np.array_equal(train_idxs, val_idxs)
    gt_idxs = np.array(
        [
            [ 0,  0],
            [ 1,  0],
            [ 0,  1],
            [ 1,  1],
            [ 0,  2],
            [ 1,  2],
            [ 0,  3],
            [ 1,  3],
            [ 0,  4],
            [ 1,  4],
            [ 0,  5],
            [ 1,  5],
            [ 0,  6],
            [ 1,  6],
            [ 0,  7],
            [ 1,  7],
            [ 0,  8],
            [ 1,  8],
            [ 0,  9],
            [ 1,  9],
            [ 0, 10],
            [ 1, 10],
            [ 0, 11],
            [ 1, 11]
        ]
    )
    assert np.array_equal(gt_idxs, train_idxs[:24])

    # Check the shapes of the output
    train_context, train_target, train_lt_vec = train_dset[0]
    val_context, val_target, val_lt_vec = val_dset[0]
    assert train_context.shape == (12, 2, 64, 64)
    assert train_target.shape == (1, 32, 32)
    assert train_lt_vec.shape == (1, 12)
    assert train_context.shape == val_context.shape
    assert train_target.shape == val_target.shape
    assert train_lt_vec.shape == val_lt_vec.shape
