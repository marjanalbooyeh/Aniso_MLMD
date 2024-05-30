import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class ParticleConfigDataset(Dataset):
    def __init__(self, traj_df):
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


class AnisoParticleDataLoader:
    def __init__(self, data_path, batch_size, overfit=False, shrink=False,
                 shrink_factor=0.1,
                 train_idx=0):
        self.data_path = data_path
        self.batch_size = batch_size
        self.overfit = overfit
        self.shrink = shrink
        self.shrink_factor = shrink_factor
        self.train_idx = train_idx

        self.train_df = pd.read_pickle(
            os.path.join(data_path, f'train_{train_idx}.pkl'))
        if self.overfit or self.shrink:
            self.train_df = self.train_df.sample(frac=self.shrink_factor).reset_index(
                drop=True)
            print("Training dataset shrunk to ", self.train_df.shape)
        self.val_df = pd.read_pickle(os.path.join(data_path, 'valid.pkl'))
        if self.shrink:
            self.val_df = self.val_df.sample(frac=self.shrink_factor*0.1).reset_index(drop=True)
            print("Validation dataset shrunk to ", self.val_df.shape)
        self.force_mean = torch.from_numpy(
            np.mean(np.asarray(list(self.train_df['force'])), axis=0))
        self.force_std = torch.from_numpy(
            np.std(np.asarray(list(self.train_df['force'])), axis=0))
        self.torque_mean = torch.from_numpy(
            np.mean(np.asarray(list(self.train_df['torque'])), axis=0))
        self.torque_std = torch.from_numpy(
            np.std(np.asarray(list(self.train_df['torque'])), axis=0))

    def get_train_dataset(self):
        train_dataset = ParticleConfigDataset(self.train_df)
        train_dataloader = _get_data_loader(dataset=train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return train_dataloader

    def get_valid_dataset(self):
        valid_dataset = ParticleConfigDataset(self.val_df)
        valid_dataloader = _get_data_loader(dataset=valid_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return valid_dataloader
