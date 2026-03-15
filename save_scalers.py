"""
Run this ONCE after training to save scaler parameters for the API.
Usage: python save_scalers.py
"""
import json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from multiprocessing import freeze_support

# Re-run the same data pipeline as battery_predictor.py to fit scalers
from battery_predictor import load_data

def main():
    csv_path = "Li_Battery_GNN_Dataset/labels.csv"
    cif_dir  = "Li_Battery_GNN_Dataset/cif"

    graphs = load_data(csv_path, cif_dir, max_samples=5000)

    scalers = {
        'voltage': StandardScaler(),
        'energy':  StandardScaler(),
        'density': StandardScaler(),
        'hull':    StandardScaler(),
    }

    import torch
    v_vals = np.array([g.y[0].item() for g in graphs])
    e_vals = np.array([g.y[1].item() for g in graphs])
    d_vals = np.array([g.y[2].item() for g in graphs])
    h_vals = np.array([g.y[3].item() for g in graphs])

    scalers['voltage'].fit(v_vals.reshape(-1,1))
    scalers['energy'].fit(e_vals.reshape(-1,1))
    scalers['density'].fit(d_vals.reshape(-1,1))
    scalers['hull'].fit(h_vals.reshape(-1,1))

    out = {}
    for key, sc in scalers.items():
        out[key] = {
            'mean':  sc.mean_.tolist(),
            'scale': sc.scale_.tolist()
        }

    with open('scalers.json', 'w') as f:
        json.dump(out, f, indent=2)

    print("✅ scalers.json saved")

if __name__ == '__main__':
    freeze_support()
    main()
