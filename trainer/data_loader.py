import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CustomTrajDataset(Dataset):
    def __init__(self, traj_df):
        self.particle_pos = torch.from_numpy(
            np.array(list(traj_df['particle_pos']))).type(torch.FloatTensor)
        self.neighbor_pos = torch.from_numpy(
            np.array(list(traj_df['neighbor_pos']))).type(torch.FloatTensor)

        self.particle_orient_q = torch.from_numpy(
            np.array(list(traj_df['particle_orient_q']))).type(
            torch.FloatTensor)
        self.neighbor_orient_q = torch.from_numpy(
            np.array(list(traj_df['neighbor_orient_q']))).type(
            torch.FloatTensor)

        self.particle_orient_R = torch.from_numpy(
            np.array(list(traj_df['particle_orient_R']))).type(
            torch.FloatTensor)
        self.neighbor_orient_R = torch.from_numpy(
            np.array(list(traj_df['neighbor_orient_R']))).type(
            torch.FloatTensor)

        self.force = torch.from_numpy(np.array(list(traj_df['force']))).type(
            torch.FloatTensor)
        self.torque = torch.from_numpy(np.array(list(traj_df['torque']))).type(
            torch.FloatTensor)
        self.energy = torch.from_numpy(np.array(list(traj_df['energy']))).type(
            torch.FloatTensor)

    def __len__(self):
        return len(self.particle_pos)

    def __getitem__(self, i):
        return (
        self.particle_pos[i], self.neighbor_pos[i], self.particle_orient_q[i],
        self.neighbor_orient_q[i],
        self.particle_orient_R[i], self.neighbor_orient_R[i]), \
            self.force[i], self.torque[i], self.energy[i]


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=5)
    return dataloader


def load_datasets(data_path, batch_size, shrink=False):
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
    if shrink:
        train_df = train_df.sample(frac=0.005).reset_index(drop=True)
        print("Training dataset shrunk to ", train_df.shape)
    val_df = pd.read_pickle(os.path.join(data_path, 'val.pkl'))
    test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    train_dataset = CustomTrajDataset(train_df)
    valid_dataset = CustomTrajDataset(val_df)
    test_dataset = CustomTrajDataset(test_df)

    train_dataloader = _get_data_loader(dataset=train_dataset,
                                        batch_size=batch_size, shuffle=True)
    valid_dataloader = _get_data_loader(dataset=valid_dataset,
                                        batch_size=batch_size, shuffle=True)
    test_dataloader = _get_data_loader(dataset=test_dataset,
                                       batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


def load_overfit_data(data_path, batch_size):
    train_df = pd.read_pickle(data_path)
    train_dataset = CustomTrajDataset(train_df)
    train_dataloader = _get_data_loader(dataset=train_dataset,
                                        batch_size=batch_size, shuffle=True)

    return train_dataloader


def load_test_dataset(data_path, batch_size):
    test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    test_dataset = CustomTrajDataset(test_df)

    test_dataloader = _get_data_loader(dataset=test_dataset,
                                       batch_size=batch_size, shuffle=False)

    return test_dataloader
