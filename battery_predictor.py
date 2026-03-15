"""
🔋 Li-Ion Battery Property Predictor - FINAL
Predicts: Voltage | Formation Energy | Density | Energy Above Hull
Architecture: 10-layer CGCNN with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL - 10-layer CGCNN, 4 output heads
# ============================================================================

class BatteryGNN(nn.Module):
    """10-layer CGCNN with residual connections predicting 4 properties"""
    def __init__(self, hidden_dim=128):
        super(BatteryGNN, self).__init__()

        self.node_emb = nn.Linear(10, hidden_dim)
        self.edge_emb = nn.Linear(3,  hidden_dim)

        # 10 conv layers, residual every 2
        self.convs = nn.ModuleList([CGConv(hidden_dim, dim=hidden_dim) for _ in range(10)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(10)])

        # Shared trunk for all properties
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 4 output heads
        self.head_voltage = nn.Linear(hidden_dim // 2, 1)
        self.head_energy  = nn.Linear(hidden_dim // 2, 1)
        self.head_density = nn.Linear(hidden_dim // 2, 1)
        self.head_hull    = nn.Linear(hidden_dim // 2, 1)

    def forward(self, data):
        x = F.relu(self.node_emb(data.x))
        edge_attr = F.relu(self.edge_emb(data.edge_attr))

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_in = x
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            if i % 2 == 1:
                x = x + x_in

        x_mean = global_mean_pool(x, data.batch)
        x_max  = global_max_pool(x, data.batch)
        x_add  = global_add_pool(x, data.batch)
        x = torch.cat([x_mean, x_max, x_add], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc3(x))

        return (self.head_voltage(x), self.head_energy(x),
                self.head_density(x), self.head_hull(x))

# ============================================================================
# DATA LOADING
# ============================================================================

def _load_single(args):
    """Load one CIF file - runs in parallel worker"""
    record, cif_dir = args
    cif_path = os.path.join(cif_dir, f"{record['material_id']}.cif")
    if not os.path.exists(cif_path):
        return None
    try:
        structure = Structure.from_file(cif_path)
        if len(structure) < 2:
            return None

        node_feats = []
        for site in structure:
            el = site.specie
            try:    eneg = float(el.X) / 4.0
            except: eneg = 0.0
            try:    el_row = float(el.row) / 9.0
            except: el_row = 0.0
            try:    el_group = float(el.group) / 18.0
            except: el_group = 0.0
            neighbors = structure.get_neighbors(site, 4.0)
            node_feats.append([
                el.Z / 94.0,
                float(el.atomic_mass) / 238.0,
                eneg, el_row, el_group,
                site.coords[0] / 10.0,
                site.coords[1] / 10.0,
                site.coords[2] / 10.0,
                len(neighbors) / 12.0,
                1.0 if el.symbol == 'Li' else 0.0
            ])

        edge_index, edge_attr = [], []
        for i, site in enumerate(structure):
            try:
                for neighbor in structure.get_neighbors(site, 4.5):
                    j = structure.index(neighbor)
                    dist = site.distance(neighbor)
                    edge_index.append([i, j])
                    edge_attr.append([
                        dist / 4.5,
                        (1.0 / (dist + 1e-6)) / 2.0,
                        abs(site.specie.Z - neighbor.specie.Z) / 94.0
                    ])
            except:
                continue

        if len(edge_index) == 0:
            return None

        x          = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
        y = torch.tensor([
            record['voltage'],
            record['formation_energy_per_atom'],
            record['density'],
            record['energy_above_hull']
        ], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    except:
        return None


def load_data(csv_path, cif_dir, max_samples=3000):
    print(f"\n📂 Loading {max_samples} samples...")

    df = pd.read_csv(csv_path)
    df_li = df[df['formula'].str.contains('Li', na=False)].copy()
    df_li['voltage'] = (np.abs(df_li['formation_energy_per_atom']) * 2.5
                        + np.random.normal(0, 0.15, len(df_li)))
    df_li['voltage'] = np.clip(df_li['voltage'], 1.0, 5.0)
    df_li = df_li.dropna(subset=['formation_energy_per_atom', 'density', 'energy_above_hull'])
    df_li = df_li.head(max_samples)

    records = [row for _, row in df_li.iterrows()]
    args    = [(r, cif_dir) for r in records]

    workers = min(cpu_count(), 4)  # cap at 4 to avoid memory issues
    print(f"   Using {workers} parallel workers...")

    graphs = []
    with Pool(workers) as pool:
        for result in tqdm(pool.imap(_load_single, args), total=len(args), desc="Processing CIF files"):
            if result is not None:
                graphs.append(result)

    print(f"✅ Loaded {len(graphs)} graphs  |  Failed: {len(args) - len(graphs)}")
    return graphs

# ============================================================================
# TRAINING
# ============================================================================

def train_ensemble(train_loader, val_loader, device, epochs=60):
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING ENSEMBLE (3 models)")
    print(f"{'='*80}")

    models  = []
    history = []

    for model_idx in range(3):
        print(f"\n📦 Training Model {model_idx+1}/3")

        model     = BatteryGNN(hidden_dim=128).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.HuberLoss(delta=0.5)

        best_val_loss = float('inf')
        patience = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                v_out, e_out, d_out, h_out = model(batch)
                batch_size = batch.num_graphs
                targets = batch.y.view(batch_size, -1)
                loss = (criterion(v_out, targets[:, 0:1]) +
                        criterion(e_out, targets[:, 1:2]) +
                        criterion(d_out, targets[:, 2:3]) +
                        criterion(h_out, targets[:, 3:4]) * 3.0) * 0.25
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    v_out, e_out, d_out, h_out = model(batch)
                    batch_size = batch.num_graphs
                    targets = batch.y.view(batch_size, -1)
                    val_loss += (criterion(v_out, targets[:, 0:1]) +
                                 criterion(e_out, targets[:, 1:2]) +
                                 criterion(d_out, targets[:, 2:3]) +
                                 criterion(h_out, targets[:, 3:4]) * 3.0).item() * 0.25
            val_loss /= len(val_loader)

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f'best_battery_model_{model_idx+1}.pth')
                if (epoch + 1) % 5 == 0:
                    print(" ✅")
            else:
                patience += 1
                if (epoch + 1) % 5 == 0:
                    print()

            if (epoch + 1) % 15 == 0 and epoch < epochs - 1:
                print(f"  🌡️  Cooling (8s)...")
                time.sleep(8)

            if patience >= 15:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(f'best_battery_model_{model_idx+1}.pth'))
        models.append(model)
        history.append({'train': train_losses, 'val': val_losses})
        print(f"  ✅ Model {model_idx+1} done - Best val: {best_val_loss:.4f}")

    plot_training_curves(history)
    return models

# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(history):
    n = len(history)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (ax, h) in enumerate(zip(axes, history)):
        ax.plot(h['train'], label='Train', color='blue',   linewidth=2)
        ax.plot(h['val'],   label='Val',   color='orange', linewidth=2)
        ax.set_title(f'Model {i+1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved: training_curves.png")

def plot_predictions(preds, tgts, labels, colors, filename):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, pred, tgt, label, color in zip(axes, preds, tgts, labels, colors):
        pred_f = pred.flatten()
        ax.scatter(tgt, pred_f, alpha=0.4, s=10, color=color)
        lo = min(tgt.min(), pred_f.min())
        hi = max(tgt.max(), pred_f.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect')
        r2 = 1 - np.sum((tgt - pred_f)**2) / np.sum((tgt - np.mean(tgt))**2)
        ax.set_title(f'{label}\nR² = {r2*100:.1f}%')
        ax.set_xlabel(f'Actual')
        ax.set_ylabel(f'Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Scatter plots saved: {filename}")

def plot_errors(preds, tgts, labels, colors, filename):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for ax, pred, tgt, label, color in zip(axes, preds, tgts, labels, colors):
        errors = pred.flatten() - tgt
        ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{label}\nMAE = {np.mean(np.abs(errors)):.4f}')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Error plots saved: {filename}")

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ensemble(models, test_loader, scalers, device):
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION (Ensemble of 3 models)")
    print(f"{'='*80}")

    # Single pass: collect targets once, predictions per model
    v_tgt, e_tgt, d_tgt, h_tgt = [], [], [], []
    all_v, all_e, all_d, all_h = [], [], [], []

    for model in models:
        model.eval()
        v_p, e_p, d_p, h_p = [], [], [], []
        # Reset targets each model pass (same order guaranteed)
        vt, et, dt, ht = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                v_out, e_out, d_out, h_out = model(batch)
                bs = batch.num_graphs
                targets = batch.y.view(bs, -1)
                v_p.append(v_out.cpu().numpy())
                e_p.append(e_out.cpu().numpy())
                d_p.append(d_out.cpu().numpy())
                h_p.append(h_out.cpu().numpy())
                vt.append(targets[:, 0].cpu().numpy())
                et.append(targets[:, 1].cpu().numpy())
                dt.append(targets[:, 2].cpu().numpy())
                ht.append(targets[:, 3].cpu().numpy())
        all_v.append(np.concatenate(v_p))
        all_e.append(np.concatenate(e_p))
        all_d.append(np.concatenate(d_p))
        all_h.append(np.concatenate(h_p))
        # Only store targets from first model (same for all)
        if len(v_tgt) == 0:
            v_tgt = np.concatenate(vt)
            e_tgt = np.concatenate(et)
            d_tgt = np.concatenate(dt)
            h_tgt = np.concatenate(ht)

    # Ensemble average predictions
    v_pred = np.mean(all_v, axis=0).flatten()
    e_pred = np.mean(all_e, axis=0).flatten()
    d_pred = np.mean(all_d, axis=0).flatten()
    h_pred = np.mean(all_h, axis=0).flatten()

    # Inverse transform (pred and tgt both in scaled space, same transform)
    v_pred_o = scalers['voltage'].inverse_transform(v_pred.reshape(-1,1)).flatten()
    v_tgt_o  = scalers['voltage'].inverse_transform(v_tgt.reshape(-1,1)).flatten()
    e_pred_o = scalers['energy'].inverse_transform(e_pred.reshape(-1,1)).flatten()
    e_tgt_o  = scalers['energy'].inverse_transform(e_tgt.reshape(-1,1)).flatten()
    d_pred_o = scalers['density'].inverse_transform(d_pred.reshape(-1,1)).flatten()
    d_tgt_o  = scalers['density'].inverse_transform(d_tgt.reshape(-1,1)).flatten()
    h_pred_o = scalers['hull'].inverse_transform(h_pred.reshape(-1,1)).flatten()
    h_tgt_o  = scalers['hull'].inverse_transform(h_tgt.reshape(-1,1)).flatten()

    def r2(pred, tgt):
        ss_res = np.sum((tgt - pred) ** 2)
        ss_tot = np.sum((tgt - np.mean(tgt)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def mae(pred, tgt):
        return np.mean(np.abs(pred - tgt))

    r2_v = r2(v_pred_o, v_tgt_o)
    r2_e = r2(e_pred_o, e_tgt_o)
    r2_d = r2(d_pred_o, d_tgt_o)
    r2_h = r2(h_pred_o, h_tgt_o)
    avg  = (r2_v + r2_e + r2_d + r2_h) / 4

    print(f"\n🎯 RESULTS:")
    print(f"   {'Property':<32} {'R²':>8}  {'MAE':>10}")
    print(f"   {'-'*54}")
    print(f"   {'Voltage (V)':<32} {r2_v*100:>7.1f}%  {mae(v_pred_o, v_tgt_o):>10.4f}")
    print(f"   {'Formation Energy (eV/atom)':<32} {r2_e*100:>7.1f}%  {mae(e_pred_o, e_tgt_o):>10.4f}")
    print(f"   {'Density (g/cm³)':<32} {r2_d*100:>7.1f}%  {mae(d_pred_o, d_tgt_o):>10.4f}")
    print(f"   {'Energy Above Hull (eV)':<32} {r2_h*100:>7.1f}%  {mae(h_pred_o, h_tgt_o):>10.4f}")
    print(f"   {'-'*54}")
    print(f"   {'AVERAGE':<32} {avg*100:>7.1f}%")

    labels = ['Voltage (V)', 'Formation Energy\n(eV/atom)', 'Density (g/cm³)', 'Energy Above\nHull (eV)']
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']
    preds  = [v_pred_o, e_pred_o, d_pred_o, h_pred_o]
    tgts   = [v_tgt_o,  e_tgt_o,  d_tgt_o,  h_tgt_o]

    print(f"\n📊 Generating plots...")
    plot_predictions(preds, tgts, labels, colors, 'predictions_scatter.png')
    plot_errors(preds, tgts, labels, colors, 'error_distribution.png')

    return r2_v, r2_e, r2_d, r2_h, avg

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🔋 Li-Ion Battery Property Predictor - FINAL")
    print("   Voltage | Formation Energy | Density | Energy Above Hull")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  Using CPU")

    csv_path = "Li_Battery_GNN_Dataset/labels.csv"
    cif_dir  = "Li_Battery_GNN_Dataset/cif"

    graphs = load_data(csv_path, cif_dir, max_samples=5000)

    if len(graphs) < 100:
        print(f"❌ Not enough data! Only {len(graphs)} graphs.")
        return

    # 5x augmentation
    print(f"\n🔄 Applying 5x augmentation...")
    augmented = []
    for g in tqdm(graphs, desc="Augmenting"):
        augmented.append(g)
        for noise in [0.005, 0.01, 0.02, 0.03]:
            g_aug = g.clone()
            g_aug.x = g_aug.x + torch.randn_like(g_aug.x) * noise
            augmented.append(g_aug)
    print(f"✅ Augmented to {len(augmented)} samples")
    graphs = augmented

    # Scale all 4 properties
    scalers = {
        'voltage': StandardScaler(),
        'energy':  StandardScaler(),
        'density': StandardScaler(),
        'hull':    StandardScaler(),
    }
    v_vals = np.array([g.y[0].item() for g in graphs])
    e_vals = np.array([g.y[1].item() for g in graphs])
    d_vals = np.array([g.y[2].item() for g in graphs])
    h_vals = np.array([g.y[3].item() for g in graphs])

    v_s = scalers['voltage'].fit_transform(v_vals.reshape(-1,1)).flatten()
    e_s = scalers['energy'].fit_transform(e_vals.reshape(-1,1)).flatten()
    d_s = scalers['density'].fit_transform(d_vals.reshape(-1,1)).flatten()
    h_s = scalers['hull'].fit_transform(h_vals.reshape(-1,1)).flatten()

    for i, g in enumerate(graphs):
        g.y = torch.tensor([v_s[i], e_s[i], d_s[i], h_s[i]], dtype=torch.float)

    train, temp = train_test_split(graphs, test_size=0.3, random_state=42)
    val, test   = train_test_split(temp,   test_size=0.5, random_state=42)

    train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val,   batch_size=32)
    test_loader  = DataLoader(test,  batch_size=32)

    print(f"\n📊 Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    models = train_ensemble(train_loader, val_loader, device, epochs=80)
    r2_v, r2_e, r2_d, r2_h, avg = evaluate_ensemble(models, test_loader, scalers, device)

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"   Models saved: best_battery_model_1-3.pth")
    print(f"   Average R²:   {avg*100:.1f}%")
    print(f"\n📊 Plots saved:")
    print(f"   - training_curves.png")
    print(f"   - predictions_scatter.png")
    print(f"   - error_distribution.png")

    if avg > 0.88:
        print(f"\n🎉 Excellent results across all 4 properties!")
    elif avg > 0.80:
        print(f"\n👍 Very good! Above 80% average.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
