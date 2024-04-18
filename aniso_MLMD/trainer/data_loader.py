import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class CustomTrajDataset(Dataset):
    def __init__(self, traj_df, force_scaler=None, torque_scaler=None):
        self.position = torch.from_numpy(
            np.array(list(traj_df['position']))).type(torch.FloatTensor)

        self.orientation_q = torch.from_numpy(
            np.array(list(traj_df['orientation_q']))).type(
            torch.FloatTensor)
        self.orientation_R = torch.from_numpy(
            np.array(list(traj_df['orientation_R']))).type(
            torch.FloatTensor)
        # self.orientation_euler = torch.from_numpy(
        #     np.array(list(traj_df['orientation_euler']))).type(
        #     torch.FloatTensor)
        self.neighbor_list = torch.from_numpy(
            np.asarray(list(traj_df['neighbor_list'])).astype(np.int64))

        self.energy = torch.from_numpy(np.array(list(traj_df['energy']))).type(
            torch.FloatTensor)

        self.force = torch.from_numpy(np.array(list(traj_df['force']))).type(
            torch.FloatTensor)
        print(force_scaler)
        if force_scaler is not None:
            self.force = force_scaler.transform(self.force)
        self.torque = torch.from_numpy(np.array(list(traj_df['torque']))).type(
            torch.FloatTensor)
        if torque_scaler is not None:
            self.torque = torque_scaler.transform(self.torque)

    def __len__(self):
        return len(self.position)

    def __getitem__(self, i):
        return (
            (
                self.position[i], self.orientation_q[i],
                self.orientation_R[i], self.neighbor_list[i]
            ),
            self.force[i], self.torque[i], self.energy[i]
        )


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=1)
    return dataloader


class MinMaxScaler:
    def __init__(self, data, scale_range=(-1, 1)):
        self.X = data.reshape(-1, 3)
        self.min = scale_range[0]
        self.max = scale_range[1]

        self.X_min = self.X.min(dim=0)[0]
        self.X_max = self.X.max(dim=0)[0]

    def transform(self, data):
        n_sample, n_beads, _ = data.shape
        data = data.reshape(-1, 3)
        X_std = (data - self.X_min) / (self.X_max - self.X_min)
        transformed_data = X_std * (self.max - self.min) + self.min
        return transformed_data.reshape(n_sample, n_beads, 3).to(data.device)

    def inv_transform(self, data):
        n_sample, n_beads, _ = data.shape
        data = data.reshape(-1, 3)
        self.X_min = self.X_min.to(data.device)
        self.X_max = self.X_max.to(data.device)
        u = ((data - self.min) * (self.X_max - self.X_min)) / (
                self.max - self.min)
        inv_transformed_data = u + self.X_min
        return inv_transformed_data.reshape(n_sample, n_beads, 3).to(
            data.device)


class AnisoDataLoader:
    def __init__(self, data_path, batch_size, shrink=False, overfit=False,
                 processor_type="MinMaxScaler", scale_range=(-1, 1)):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shrink = shrink
        self.overfit = overfit
        self.processor_type = processor_type

        self.train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
        self.val_df = pd.read_pickle(os.path.join(data_path, 'val.pkl'))
        self.test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))

        self.force_scaler = None
        self.torque_scaler = None

        # create force and torque scalers
        if self.processor_type == "MinMaxScaler":
            self.train_force = torch.from_numpy(
                np.array(list(self.train_df['force']))).type(
                torch.FloatTensor)
            self.force_scaler = MinMaxScaler(self.train_force,
                                             scale_range=scale_range)

            self.train_torque = torch.from_numpy(
                np.array(list(self.train_df['torque']))).type(
                torch.FloatTensor)
            self.torque_scaler = MinMaxScaler(self.train_torque,
                                              scale_range=scale_range)

    def get_train_dataset(self):
        train_dataset = CustomTrajDataset(self.train_df, self.force_scaler,
                                          self.torque_scaler)
        train_dataloader = _get_data_loader(dataset=train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return train_dataloader

    def get_valid_dataset(self):
        valid_dataset = CustomTrajDataset(self.val_df, self.force_scaler,
                                          self.torque_scaler)
        valid_dataloader = _get_data_loader(dataset=valid_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return valid_dataloader

    def get_test_dataset(self):
        test_dataset = CustomTrajDataset(self.test_df, self.force_scaler,
                                         self.torque_scaler)
        test_dataloader = _get_data_loader(dataset=test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)
        return test_dataloader


def load_datasets(data_path, batch_size, shrink=False):
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
    if shrink:
        train_df = train_df.sample(frac=0.1).reset_index(drop=True)
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
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
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
