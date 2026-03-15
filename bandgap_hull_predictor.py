"""
⚡ Energy Above Hull Predictor
Focused single-property predictor with deep architecture
Target: 85%+ R²
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL - Deep 10-layer CGCNN focused on energy above hull
# ============================================================================

class HullGNN(nn.Module):
    """10-layer CGCNN - stable config for energy above hull"""
    def __init__(self, hidden_dim=128):
        super(HullGNN, self).__init__()

        self.node_emb = nn.Linear(10, hidden_dim)
        self.edge_emb = nn.Linear(3,  hidden_dim)

        # 10 conv layers with residual skip every 2
        self.convs = nn.ModuleList([CGConv(hidden_dim, dim=hidden_dim) for _ in range(10)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(10)])

        # Simple stable MLP head (no huge expansion)
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out  = nn.Linear(hidden_dim // 2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

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
        return self.out(x)

# ============================================================================
# DATA LOADING - richer features
# ============================================================================

def load_data(csv_path, cif_dir, max_samples=3000):
    print(f"\n📂 Loading {max_samples} samples...")

    df = pd.read_csv(csv_path)
    df_li = df[df['formula'].str.contains('Li', na=False)].copy()
    df_li = df_li.dropna(subset=['energy_above_hull'])

    if max_samples:
        df_li = df_li.head(max_samples)

    graphs = []
    failed = 0
    pbar = tqdm(df_li.iterrows(), total=len(df_li), desc="Processing CIF files")

    for idx, record in pbar:
        cif_path = os.path.join(cif_dir, f"{record['material_id']}.cif")
        if not os.path.exists(cif_path):
            failed += 1
            continue

        try:
            structure = Structure.from_file(cif_path)
            if len(structure) < 2:
                continue

            # Richer node features (10 features)
            node_feats = []
            for site in structure:
                el = site.specie
                neighbors = structure.get_neighbors(site, 4.0)
                try:
                    eneg = float(el.X) / 4.0
                except Exception:
                    eneg = 0.0
                try:
                    el_row = float(el.row) / 9.0
                except Exception:
                    el_row = 0.0
                try:
                    el_group = float(el.group) / 18.0
                except Exception:
                    el_group = 0.0
                node_feats.append([
                    el.Z / 94.0,
                    float(el.atomic_mass) / 238.0,
                    eneg,
                    el_row,
                    el_group,
                    site.coords[0] / 10.0,
                    site.coords[1] / 10.0,
                    site.coords[2] / 10.0,
                    len(neighbors) / 12.0,
                    1.0 if el.symbol == 'Li' else 0.0
                ])

            # Richer edge features (3 features)
            edge_index, edge_attr = [], []
            for i, site in enumerate(structure):
                try:
                    neighbors = structure.get_neighbors(site, 4.5)
                    for neighbor in neighbors:
                        j = structure.index(neighbor)
                        dist = site.distance(neighbor)
                        bond_order = 1.0 / (dist + 1e-6)
                        edge_index.append([i, j])
                        edge_attr.append([
                            dist / 4.5,
                            bond_order / 2.0,
                            abs(site.specie.Z - neighbor.specie.Z) / 94.0
                        ])
                except:
                    continue

            if len(edge_index) == 0:
                continue

            x          = torch.tensor(node_feats, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
            y          = torch.tensor([record['energy_above_hull']], dtype=torch.float)

            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
            break
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n⚠️  Error {record['material_id']}: {str(e)[:80]}")
            continue

    pbar.close()
    print(f"✅ Loaded {len(graphs)} graphs  |  Failed: {failed}")
    return graphs

# ============================================================================
# TRAINING
# ============================================================================

def train_ensemble(train_loader, val_loader, device, epochs=80):
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING ENSEMBLE (5 models)")
    print(f"{'='*80}")

    models  = []
    history = []

    for model_idx in range(5):
        print(f"\n📦 Training Model {model_idx+1}/5")

        model     = HullGNN(hidden_dim=128).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.HuberLoss(delta=0.5)  # robust to outliers

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
                out = model(batch)
                batch_size = batch.num_graphs
                targets = batch.y.view(batch_size, 1)
                loss = criterion(out, targets)
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
                    out = model(batch)
                    batch_size = batch.num_graphs
                    targets = batch.y.view(batch_size, 1)
                    val_loss += criterion(out, targets).item()
            val_loss /= len(val_loader)

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f'best_hull_model_{model_idx+1}.pth')
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

        model.load_state_dict(torch.load(f'best_hull_model_{model_idx+1}.pth'))
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
        ax.set_title(f'Model {i+1} - Training Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hull_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved: hull_training_curves.png")

def plot_predictions(hull_pred, hull_tgt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(hull_tgt, hull_pred, alpha=0.4, s=10, color='darkorange')
    lo, hi = min(hull_tgt.min(), hull_pred.min()), max(hull_tgt.max(), hull_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect')
    r2 = 1 - np.sum((hull_tgt - hull_pred)**2) / np.sum((hull_tgt - np.mean(hull_tgt))**2)
    ax.set_title(f'Energy Above Hull\nR² = {r2*100:.1f}%')
    ax.set_xlabel('Actual (eV)')
    ax.set_ylabel('Predicted (eV)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1]
    errors = hull_pred - hull_tgt
    ax.hist(errors, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'Error Distribution\nMAE = {np.mean(np.abs(errors)):.4f} eV')
    ax.set_xlabel('Prediction Error (eV)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hull_predictions.png', dpi=150, bbox_inches='tight')
    print(f"📊 Prediction plots saved: hull_predictions.png")

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ensemble(models, test_loader, scaler, device):
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION (Ensemble of 5 models)")
    print(f"{'='*80}")

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                preds.append(out.cpu().numpy())
        all_preds.append(np.concatenate(preds))

    hull_pred = np.mean(all_preds, axis=0).flatten()

    hull_tgt = []
    for batch in test_loader:
        batch_size = batch.num_graphs
        hull_tgt.append(batch.y.view(batch_size, 1).numpy())
    hull_tgt = np.concatenate(hull_tgt).flatten()

    # Inverse transform
    hull_pred_orig = scaler.inverse_transform(hull_pred.reshape(-1, 1)).flatten()
    hull_tgt_orig  = scaler.inverse_transform(hull_tgt.reshape(-1, 1)).flatten()

    r2  = 1 - np.sum((hull_tgt_orig - hull_pred_orig)**2) / np.sum((hull_tgt_orig - np.mean(hull_tgt_orig))**2)
    mae = np.mean(np.abs(hull_pred_orig - hull_tgt_orig))

    print(f"\n🎯 RESULTS:")
    print(f"   {'Property':<30} {'R²':>8}  {'MAE':>10}")
    print(f"   {'-'*52}")
    print(f"   {'Energy Above Hull (eV)':<30} {r2*100:>7.1f}%  {mae:>10.4f}")

    print(f"\n📊 Generating plots...")
    plot_predictions(hull_pred_orig, hull_tgt_orig)

    return r2, mae

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("⚡ Energy Above Hull Predictor")
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

    # 5x augmentation - varied noise levels
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

    # Scale
    scaler   = StandardScaler()
    hull_vals = np.array([g.y[0].item() for g in graphs])
    hull_scaled = scaler.fit_transform(hull_vals.reshape(-1, 1)).flatten()
    for i, g in enumerate(graphs):
        g.y = torch.tensor([hull_scaled[i]], dtype=torch.float)

    # Split
    train, temp = train_test_split(graphs, test_size=0.3, random_state=42)
    val, test   = train_test_split(temp,   test_size=0.5, random_state=42)

    train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val,   batch_size=32)
    test_loader  = DataLoader(test,  batch_size=32)

    print(f"\n📊 Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    models = train_ensemble(train_loader, val_loader, device, epochs=100)
    r2, mae = evaluate_ensemble(models, test_loader, scaler, device)

    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"{'='*80}")
    print(f"   Energy Above Hull R²: {r2*100:.1f}%")
    print(f"   MAE:                  {mae:.4f} eV")
    print(f"\n📊 Plots saved:")
    print(f"   - hull_training_curves.png")
    print(f"   - hull_predictions.png")
    print(f"   Models saved: best_hull_model_1-5.pth")

    if r2 > 0.85:
        print(f"\n🎉 Excellent! Ready to merge into voltage_predictor.py")
    elif r2 > 0.70:
        print(f"\n� Good. Can merge into voltage_predictor.py")
    else:
        print(f"\n⚠️  Needs more work")

if __name__ == "__main__":
    main()
