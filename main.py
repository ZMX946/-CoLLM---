import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.collm import CoLLM
from models.small import SmallModel
from models.gpt2_ts import GPT2TimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection
from datasets.cmapss import CMAPSSDataset


# ============================================================
# Config
# ============================================================

DEVICE =  'cpu'
DATA_PATH = './data/CMAPSS/train_FD001.txt'
BATCH_SIZE = 64

TAU1 = 0.6
TAU2 = 0.05

SAVE_DIR = './results'
DPI = 600

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Load Models
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
# Load Dataset
# ============================================================

dataset = CMAPSSDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Inference
# ============================================================

ys_list, yl_list, yc_list, ytrue_list = [], [], [], []
Qs_list, Ql_list = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)

        ys, phi_s = S(x)
        yl, phi_l = L(x)
        yc = model.inference(x, TAU1, TAU2)

        Qs = Fz(phi_s)
        Ql = Rf(phi_l)

        ys_list.append(ys.cpu().numpy())
        yl_list.append(yl.cpu().numpy())
        yc_list.append(yc.cpu().numpy())
        ytrue_list.append(y.numpy())

        Qs_list.append(Qs.cpu().numpy())
        Ql_list.append(Ql.cpu().numpy())


ys = np.concatenate(ys_list)
yl = np.concatenate(yl_list)
yc = np.concatenate(yc_list)
ytrue = np.concatenate(ytrue_list)
Qs = np.concatenate(Qs_list)
Ql = np.concatenate(Ql_list)


# ============================================================
# Metrics
# ============================================================

def rmse(p, y):
    return np.sqrt(np.mean((p - y) ** 2))

print(f'RMSE Small : {rmse(ys, ytrue):.3f}')
print(f'RMSE Large : {rmse(yl, ytrue):.3f}')
print(f'RMSE CoLLM : {rmse(yc, ytrue):.3f}')


# ============================================================
# Plot 1: RUL Prediction Comparison
# ============================================================

N_SHOW = 300

plt.figure(figsize=(10, 4))
plt.plot(ytrue[:N_SHOW], label='Ground Truth', linewidth=2)
plt.plot(ys[:N_SHOW], '--', label='Small Model')
plt.plot(yl[:N_SHOW], ':', label='Large Model')
plt.plot(yc[:N_SHOW], label='CoLLM', linewidth=2)

plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('RUL Prediction Comparison (FD001)')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/rul_comparison.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/rul_comparison.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 2: Error Distribution
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
plt.title('Error Distribution')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/error_distribution.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/error_distribution.pdf', dpi=DPI)
plt.close()


# ============================================================
# Plot 3: Confidence Routing Behavior
# ============================================================

plt.figure(figsize=(6, 5))
plt.scatter(Qs, Ql, s=5, alpha=0.5)
plt.axvline(TAU1, color='r', linestyle='--', label=r'$\tau_1$')
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')

plt.xlabel(r'$Q_s$ (Small Confidence)')
plt.ylabel(r'$Q_l$ (Large Confidence)')
plt.title('Confidence Routing Behavior')
plt.legend()
plt.tight_layout()

plt.savefig(f'{SAVE_DIR}/confidence_routing.png', dpi=DPI)
plt.savefig(f'{SAVE_DIR}/confidence_routing.pdf', dpi=DPI)
plt.close()


print(f'\nAll figures saved to ./{SAVE_DIR}/ (PNG + PDF, dpi={DPI})')
