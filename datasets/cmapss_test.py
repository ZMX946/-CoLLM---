import numpy as np
import torch
from torch.utils.data import Dataset

class CMAPSSTestDataset(Dataset):
    def __init__(self, test_path, rul_path, window_size=50):
        data = np.loadtxt(test_path)
        rul_last = np.loadtxt(rul_path)

        unit_ids = data[:, 0].astype(int)
        cycles = data[:, 1]
        SENSOR_IDX = [
            1, 2, 3, 6, 7, 8,
            10, 11, 12, 13, 14, 16,
            19, 20
        ]

        # data[:, 5:] 是 21 sensors，从 0-based 转换
        sensors = data[:, [5 + i for i in SENSOR_IDX]]

        sensors = (sensors - sensors.mean(0)) / (sensors.std(0) + 1e-6)

        self.X, self.y = [], []

        for uid in np.unique(unit_ids):
            idx = np.where(unit_ids == uid)[0]
            if len(idx) < window_size:
                continue

            x = sensors[idx[-window_size:]]
            self.X.append(x)
            self.y.append(rul_last[uid - 1])

        self.X = torch.from_numpy(np.array(self.X)).float()
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
