import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from models.collm import CoLLM
from models.small import SmallModel
from models.gpt2_ts import GPT2TimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss_test import CMAPSSTestDataset


# ============================================================
# Config
# ============================================================

DEVICE = 'cpu'
DATA_ROOT = 'data/CMAPSS'
BATCH_SIZE = 64
DPI = 600
SAVE_DIR = './results_test'
N_SHOW = 300

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Load models
# ============================================================

S = SmallModel().to(DEVICE)
S.load_state_dict(torch.load('./train/small.pt', map_location=DEVICE))
S.eval()

L = GPT2TimeSeries().to(DEVICE)
L.load_state_dict(torch.load('./train/large.pt', map_location=DEVICE))
L.eval()

Fz = FuzzyDecisionAgent(32, 50).to(DEVICE)
Fz.load_state_dict(torch.load('./train/fuzzy.pt', map_location=DEVICE))
Fz.eval()

Rf = SelfReflection(768, 12).to(DEVICE)
Rf.load_state_dict(torch.load('./train/reflect.pt', map_location=DEVICE))
Rf.eval()

model = CoLLM(S, L, Fz, Rf)


# ============================================================
# Load TEST dataset
# ============================================================

dataset = CMAPSSTestDataset(
    f'{DATA_ROOT}/test_FD001.txt',
    f'{DATA_ROOT}/RUL_FD001.txt'
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Evaluation
# ============================================================

ys, yl, yc, ytrue = [], [], [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)

        # Small
        y_s, _ = S(x)

        # Large
        y_l, _ = L(x)

        # CoLLM
        y_c = model.inference(x)

        ys.append(y_s.numpy())
        yl.append(y_l.numpy())
        yc.append(y_c.numpy())
        ytrue.append(y.numpy())

ys = np.concatenate(ys)
yl = np.concatenate(yl)
yc = np.concatenate(yc)
ytrue = np.concatenate(ytrue)


def rmse(p, y):
    return np.sqrt(np.mean((p - y) ** 2))


print(f'RMSE Small : {rmse(ys, ytrue):.3f}')
print(f'RMSE Large : {rmse(yl, ytrue):.3f}')
print(f'RMSE CoLLM : {rmse(yc, ytrue):.3f}')


# ============================================================
# Plot 1: RUL Prediction Comparison (Test)
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(ytrue[:N_SHOW], label='Ground Truth', linewidth=2)
plt.plot(ys[:N_SHOW], '--', label='Small Model')
plt.plot(yl[:N_SHOW], ':', label='Large Model')
plt.plot(yc[:N_SHOW], label='CoLLM', linewidth=2)

plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('RUL Prediction on Test Set (FD001)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/test_rul_comparison.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/test_rul_comparison.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 2: Error Distribution (Test)
# ============================================================

err_s = ys - ytrue
err_l = yl - ytrue
err_c = yc - ytrue

plt.figure(figsize=(6, 4))
plt.hist(err_s, bins=50, alpha=0.5, label='Small')
plt.hist(err_l, bins=50, alpha=0.5, label='Large')
plt.hist(err_c, bins=50, alpha=0.7, label='CoLLM')

plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution on Test Set (FD001)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/test_error_distribution.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/test_error_distribution.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 3: Error vs RUL (Test)
# ============================================================

plt.figure(figsize=(6, 4))
plt.scatter(ytrue, err_s, s=5, alpha=0.3, label='Small')
plt.scatter(ytrue, err_l, s=5, alpha=0.3, label='Large')
plt.scatter(ytrue, err_c, s=5, alpha=0.4, label='CoLLM')
plt.axhline(0, linestyle='--')

plt.xlabel('Ground Truth RUL')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs RUL (Test Set)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/test_error_vs_rul.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/test_error_vs_rul.pdf', dpi=DPI)
plt.close()
