import numpy as np
import torch
from torch.utils.data import Dataset


class CMAPSSDataset(Dataset):
    def __init__(self, data_path, window_size=50, stride=1):
        data = np.loadtxt(data_path)
        unit_ids = data[:, 0].astype(int)
        cycles = data[:, 1]
        # CMAPSS standard 14 sensors (1-based index in paper)
        SENSOR_IDX = [
            1, 2, 3, 6, 7, 8,
            10, 11, 12, 13, 14, 16,
            19, 20
        ]

        # data[:, 5:] 是 21 sensors，从 0-based 转换
        sensors = data[:, [5 + i for i in SENSOR_IDX]]

        rul = []
        for uid in np.unique(unit_ids):
            idx = unit_ids == uid
            max_cycle = cycles[idx].max()
            rul.extend(max_cycle - cycles[idx])
        rul = np.array(rul)


        sensors = (sensors - sensors.mean(0)) / (sensors.std(0) + 1e-6)


        self.X, self.y = [], []
        for uid in np.unique(unit_ids):
            idx = np.where(unit_ids == uid)[0]
            for i in range(0, len(idx) - window_size + 1, stride):
                self.X.append(sensors[idx[i:i+window_size]])
                self.y.append(rul[idx[i+window_size-1]])


        self.X = torch.from_numpy(np.array(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
