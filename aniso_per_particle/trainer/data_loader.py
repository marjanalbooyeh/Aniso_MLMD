import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class ParticleConfigDataset(Dataset):
    def __init__(self, traj_df, force_scaler=None, torque_scaler=None):
        self.dr = torch.from_numpy(
            np.array(list(traj_df['dr']))).type(torch.FloatTensor)

        self.orientation = torch.from_numpy(
            np.array(list(traj_df['orientation']))).type(
            torch.FloatTensor)
        self.n_orientation = torch.from_numpy(
            np.array(list(traj_df['n_orientation']))).type(
            torch.FloatTensor)

        self.energy = torch.from_numpy(np.array(list(traj_df['energy']))).type(
            torch.FloatTensor)

        self.force = torch.from_numpy(np.array(list(traj_df['force']))).type(
            torch.FloatTensor)

        self.torque = torch.from_numpy(np.array(list(traj_df['torque']))).type(
            torch.FloatTensor)
        if force_scaler is not None:
            print('force_scaler.transform')
            self.force = force_scaler.transform(self.force)
        if torque_scaler is not None:
            self.torque = torque_scaler.transform(self.torque)
            print('torque_scaler.transform')

    def __len__(self):
        return len(self.dr)

    def __getitem__(self, i):
        return (
            (
                self.dr[i], self.orientation[i],
                self.n_orientation[i]
            ),
            self.force[i], self.torque[i], self.energy[i]
        )


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=0)
    return dataloader
class MinMaxScaler:
    def __init__(self, data, scale_range=(0, 1)):
        self.X = data.reshape(-1, 3)
        self.min = scale_range[0]
        self.max = scale_range[1]

        self.X_min = self.X.min(dim=0)[0]
        self.X_max = self.X.max(dim=0)[0]

    def transform(self, data):
        n_sample, _ = data.shape
        data = data.reshape(-1, 3)
        X_std = (data - self.X_min) / (self.X_max - self.X_min)
        transformed_data = X_std * (self.max - self.min) + self.min
        return transformed_data.reshape(n_sample, 3).to(data.device)

    def inv_transform(self, data):
        n_sample, _ = data.shape
        data = data.reshape(-1, 3)
        self.X_min = self.X_min.to(data.device)
        self.X_max = self.X_max.to(data.device)
        u = ((data - self.min) * (self.X_max - self.X_min)) / (
                self.max - self.min)
        inv_transformed_data = u + self.X_min
        return inv_transformed_data.reshape(n_sample, 3).to(
            data.device)


class AnisoParticleDataLoader:
    def __init__(self, data_path, batch_size, overfit=False, shrink=False,
                 shrink_factor=0.1, train_idx=0, processor_type="MinMaxScaler", scale_range=(0, 1)):
        self.data_path = data_path
        self.batch_size = batch_size
        self.overfit = overfit
        self.shrink = shrink
        self.shrink_factor = shrink_factor
        self.train_idx = train_idx
        self.processor_type = processor_type


        self.train_df = pd.read_pickle(
            os.path.join(data_path, f'train_{train_idx}.pkl'))
        if self.overfit or self.shrink:
            self.train_df = self.train_df.sample(frac=self.shrink_factor).reset_index(
                drop=True)
            print("Training dataset shrunk to ", self.train_df.shape)
        if not self.overfit:
            self.val_df = pd.read_pickle(os.path.join(data_path, 'valid.pkl'))
            if self.shrink:
                self.val_df = self.val_df.sample(frac=self.shrink_factor).reset_index(drop=True)
                print("Validation dataset shrunk to ", self.val_df.shape)
        self.force_scaler = None
        self.torque_scaler = None
        # create force and torque scalers
        if self.processor_type == "MinMaxScaler":
            print(self.processor_type)
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
        train_dataset = ParticleConfigDataset(self.train_df, self.force_scaler, self.torque_scaler)
        train_dataloader = _get_data_loader(dataset=train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return train_dataloader

    def get_valid_dataset(self):
        valid_dataset = ParticleConfigDataset(self.val_df, self.force_scaler, self.torque_scaler)
        valid_dataloader = _get_data_loader(dataset=valid_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return valid_dataloader
