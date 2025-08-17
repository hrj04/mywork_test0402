# ============================
# 0) Imports & config
# ============================
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# User inputs / knobs
# ----------------------------
EMBED_DIM        = 16     # AE bottleneck size (try 8/16/32)
EPOCHS           = 200
BATCH_SIZE       = 64
LR               = 1e-3
L1_LATENT_LAMBDA = 1e-4   # sparsity penalty on latent z
DENOISE_NOISE    = 0.05   # Gaussian noise std for denoising
N_CLUSTERS       = 30     # KMeans clusters (grid search or silhouette to choose)
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 1) Data: splits & scaling
# ============================
# R: DataFrame (dates × alphas) with daily pnl (or IC)
# Example:
# R = pd.read_parquet("alpha_pnl_matrix.parquet")  # index: dates, columns: alpha IDs

assert isinstance(R, pd.DataFrame) and R.index.is_monotonic_increasing

T = len(R)
t1 = int(T * 0.60)   # train
t2 = int(T * 0.80)   # validation
R_train, R_val, R_test = R.iloc[:t1], R.iloc[t1:t2], R.iloc[t2:]

# Standardize per alpha using train stats only (mean/std across time)
scaler = StandardScaler()
R_train_z = pd.DataFrame(scaler.fit_transform(R_train), index=R_train.index, columns=R_train.columns)
R_val_z   = pd.DataFrame(scaler.transform(R_val),   index=R_val.index,   columns=R_val.columns)
R_test_z  = pd.DataFrame(scaler.transform(R_test),  index=R_test.index,  columns=R_test.columns)

# ============================
# 2) Dataset: "one sample = one alpha time-series"
# ============================
class AlphaSeriesDataset(Dataset):
    def __init__(self, Rz: pd.DataFrame, denoise_std=0.0):
        # X: N × T  (transpose: alphas as rows)
        self.X = torch.tensor(Rz.values.T, dtype=torch.float32)
        self.denoise_std = denoise_std
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = self.X[i]  # shape: T
        if self.denoise_std > 0:
            x_noisy = x + self.denoise_std * torch.randn_like(x)
        else:
            x_noisy = x
        return x_noisy, x

train_ds = AlphaSeriesDataset(R_train_z, denoise_std=DENOISE_NOISE)
val_ds   = AlphaSeriesDataset(R_val_z,   denoise_std=0.0)  # monitor with clean val
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ============================
# 3) Sparse Denoising Autoencoder
# ============================
class SparseAE(nn.Module):
    def __init__(self, T_in, d=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(T_in, 256), nn.ReLU(),
            nn.Linear(256, 64),   nn.ReLU(),
            nn.Linear(64, d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, 64),   nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, T_in)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

T_in = R_train_z.shape[0]  # number of train days
model = SparseAE(T_in, d=EMBED_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
mse = nn.MSELoss()

def train_ae(model, train_loader, val_loader, epochs, l1_lambda, device):
    best_state, best_val = None, float("inf")
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for x_noisy, x in train_loader:
            x_noisy, x = x_noisy.to(device), x.to(device)
            opt.zero_grad()
            x_hat, z = model(x_noisy)
            loss = mse(x_hat, x) + l1_lambda * torch.mean(torch.abs(z))
            loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)

        # validation (reconstruct val)
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for x_noisy, x in val_loader:
                x_noisy, x = x_noisy.to(device), x.to(device)
                x_hat, z = model(x_noisy)
                vloss += mse(x_hat, x).item() * x.size(0)
            vloss /= len(val_loader.dataset)

        if vloss < best_val:
            best_val, best_state = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0:
            print(f"[AE] epoch {ep:04d}  train {tr_loss:.4f}  val {vloss:.4f}")

    model.load_state_dict(best_state)
    return model

model = train_ae(model, train_loader, val_loader, EPOCHS, L1_LATENT_LAMBDA, DEVICE)

# ============================
# 4) Get per-alpha embeddings (latent z)
#    IMPORTANT: fit embeddings on train+val (no test leakage into training!)
# ============================
model.eval()
with torch.no_grad():
    X_all = torch.tensor(pd.concat([R_train_z, R_val_z], axis=0).values.T, dtype=torch.float32).to(DEVICE)  # N × (Ttrain+Tval)
    # Our encoder was trained with input dim T_in (train-only length).
    # To avoid shape mismatch, embed using TRAIN-only portion for each alpha:
    X_train_only = torch.tensor(R_train_z.values.T, dtype=torch.float32).to(DEVICE)
    _, Z = model(X_train_only)  # N × d
Z = Z.cpu().numpy()
embeddings_df = pd.DataFrame(Z, index=R.columns, columns=[f"z{j+1}" for j in range(EMBED_DIM)])

# ============================
# 5) Cluster embeddings & pick representatives
# ============================
# (Optionally choose N_CLUSTERS via a small grid on val silhouette)
def pick_k_by_silhouette(Z, k_list=(10, 20, 30, 40, 50)):
    best_k, best_s = None, -1
    for k in k_list:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Z)
        s = silhouette_score(Z, km.labels_) if k > 1 else -1
        if s > best_s:
            best_s, best_k = s, k
    return best_k

# N_CLUSTERS = pick_k_by_silhouette(Z) or set manually
km = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=42).fit(Z)
cluster_labels = pd.Series(km.labels_, index=R.columns, name="cluster")

def oos_sharpe(alpha_col: str) -> float:
    ser = R_test_z[alpha_col].dropna()
    if ser.std() == 0: return -9e9
    return float(ser.mean() / (ser.std() + 1e-8))

clean_alpha_list = []
for c, grp in cluster_labels.groupby(cluster_labels):
    alphas_in_cluster = grp.index.tolist()
    # champion by OOS Sharpe (or use IC mean, your metric)
    champ = max(alphas_in_cluster, key=oos_sharpe)
    clean_alpha_list.append(champ)

clean_alpha_list = pd.Index(clean_alpha_list, name="clean_pool")

# ============================
# 6) Diagnostics: did we help?
# ============================
def participation_ratio(cov):
    w, _ = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    return (w.sum()**2) / (np.square(w).sum())

orig_cov = np.cov(R_test_z.values.T)
new_cov  = np.cov(R_test_z[clean_alpha_list].values.T)

print("Effective dimensionality (orig):", participation_ratio(orig_cov))
print("Effective dimensionality (new): ", participation_ratio(new_cov))
print("Avg OOS Sharpe (orig, mean over alphas):", (R_test_z.mean()/R_test_z.std()).mean())
print("Avg OOS Sharpe (clean pool):            ", (R_test_z[clean_alpha_list].mean()/R_test_z[clean_alpha_list].std()).mean())

# Optional: save outputs
# embeddings_df.to_parquet("alpha_embeddings.parquet")
# cluster_labels.to_frame().to_csv("alpha_clusters.csv")
# pd.Series(clean_alpha_list).to_csv("alpha_clean_pool.csv", index=False)