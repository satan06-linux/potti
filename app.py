"""
🔋 Li-Ion Battery Prediction API
Flask backend serving the GNN prediction model
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import os
import sqlite3
import hashlib
import json
import tempfile
import pandas as pd
from datetime import datetime
from functools import wraps

# GNN imports
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from pymatgen.core import Structure, Composition, Lattice
from pymatgen.core.structure import Structure as PmgStructure
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# ============================================================================
# MODEL (same as battery_predictor.py)
# ============================================================================

class BatteryGNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super(BatteryGNN, self).__init__()
        self.node_emb = nn.Linear(10, hidden_dim)
        self.edge_emb = nn.Linear(3,  hidden_dim)
        self.convs = nn.ModuleList([CGConv(hidden_dim, dim=hidden_dim) for _ in range(10)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(10)])
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
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
# LOAD MODELS
# ============================================================================

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
models  = []
scalers = {}

def load_models():
    global models, scalers
    print(f"🔋 Loading models on {device}...")
    for i in range(1, 4):
        path = f'best_battery_model_{i}.pth'
        if os.path.exists(path):
            m = BatteryGNN(hidden_dim=128).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
            print(f"  ✅ Loaded model {i}")
    # Load scalers from saved file if exists, else use defaults
    if os.path.exists('scalers.json'):
        with open('scalers.json') as f:
            d = json.load(f)
        for key in ['voltage', 'energy', 'density', 'hull']:
            sc = StandardScaler()
            sc.mean_  = np.array(d[key]['mean'])
            sc.scale_ = np.array(d[key]['scale'])
            sc.var_   = sc.scale_ ** 2
            sc.n_features_in_ = 1
            scalers[key] = sc
        print("  ✅ Loaded scalers")
    else:
        print("  ⚠️  No scalers.json found - predictions will be in scaled space")
    print(f"✅ {len(models)} models ready")

# ============================================================================
# CSV INDEX - for composition lookup
# ============================================================================

csv_index = {}  # formula → material_id

def load_csv_index():
    global csv_index
    csv_path = 'Li_Battery_GNN_Dataset/labels.csv'
    if not os.path.exists(csv_path):
        print("  ⚠️  labels.csv not found - composition lookup disabled")
        return
    df = pd.read_csv(csv_path)
    # Index by reduced formula (case-insensitive)
    for _, row in df.iterrows():
        try:
            comp = Composition(row['formula'])
            key  = comp.reduced_formula.lower()
            csv_index[key] = row['material_id']
        except:
            continue
    print(f"  ✅ CSV index loaded: {len(csv_index)} materials")

# ============================================================================
# COMPOSITION → STRUCTURE (synthetic fallback)
# ============================================================================

LI_ION_ELEMENTS = {
    'Li','Co','Ni','Mn','Fe','P','O','S','F','Si','Ti','V','Cr',
    'Cu','Zn','Al','Mg','Ca','Na','K','C','N','Cl','Br','I'
}

# Elements that must be present alongside Li for a viable battery cathode
ACTIVE_ELEMENTS = {'Co','Ni','Mn','Fe','V','Ti','Cr','Cu','Si','P'}

# Known good battery formulas for suggestion matching
KNOWN_BATTERY_MATERIALS = [
    'LiCoO2','LiFePO4','LiNiO2','LiMn2O4','Li2MnO3','LiNiMnCoO2',
    'LiTiO2','LiV2O5','LiFeSiO4','Li2FeSiO4','LiCrO2','LiNi0.5Mn1.5O4',
    'LiNi0.8Co0.15Al0.05O2','LiNi0.6Mn0.2Co0.2O2','LiFe2O4',
    'LiMnO2','LiFeO2','LiVPO4F','LiCoPO4','LiMnPO4','LiNiPO4',
]

def get_battery_suggestions(comp: Composition) -> list:
    """Given a composition, suggest real battery materials with similar elements."""
    user_elements = {str(el) for el in comp.elements}
    suggestions = []
    for formula in KNOWN_BATTERY_MATERIALS:
        try:
            c = Composition(formula)
            mat_elements = {str(el) for el in c.elements}
            # Score: how many elements overlap
            overlap = len(user_elements & mat_elements)
            if overlap >= 1:
                suggestions.append((overlap, formula))
        except:
            continue
    # Sort by overlap descending, return top 5
    suggestions.sort(key=lambda x: -x[0])
    return [f for _, f in suggestions[:5]]

def is_viable_battery_material(comp: Composition) -> tuple:
    """
    Returns (is_valid: bool, reason: str)
    Checks beyond element membership — needs active TM + oxygen/phosphate framework.
    """
    elements = {str(el) for el in comp.elements}

    # Must have Li
    if 'Li' not in elements:
        return False, "No Lithium present"

    # Must have at least one electrochemically active transition metal
    active = elements & ACTIVE_ELEMENTS
    if not active:
        return False, (
            f"No electrochemically active transition metal found in '{comp.reduced_formula}'. "
            "Battery cathodes require elements like Co, Ni, Mn, Fe, V, Ti, etc."
        )

    # Must have an anion framework (O, S with active metal, F, or P+O)
    has_oxygen    = 'O' in elements
    has_phosphate = 'P' in elements and 'O' in elements
    has_sulfide   = 'S' in elements and active  # sulfide batteries exist but need TM
    has_fluoride  = 'F' in elements

    if not (has_oxygen or has_phosphate or has_sulfide or has_fluoride):
        return False, (
            f"'{comp.reduced_formula}' lacks an anion framework (O, S, F, or PO4). "
            "Battery materials need an oxide, phosphate, sulfide, or fluoride structure."
        )

    # Li:TM ratio sanity check — should be between 0.1 and 4
    li_amt = comp['Li']
    tm_amt = sum(comp[el] for el in comp.elements if str(el) in ACTIVE_ELEMENTS)
    if tm_amt > 0:
        ratio = li_amt / tm_amt
        if ratio > 6:
            return False, (
                f"Li:{active} ratio of {ratio:.1f} is too high for a practical battery material."
            )

    return True, "ok"

def build_synthetic_structure(comp: Composition) -> Structure:
    """
    Build a rocksalt-like structure from composition.
    Handles fractional occupancies (e.g. LiNi0.8Co0.1Al0.1O2) by scaling
    amounts to integers via the smallest integer multiplier.
    """
    # Scale fractional amounts to integers
    amounts_raw = {str(el): comp[el] for el in comp.elements}
    # Find multiplier to make all amounts >= 1 integer
    min_amt = min(v for v in amounts_raw.values() if v > 0)
    scale   = max(1, round(1.0 / min_amt)) if min_amt < 1 else 1
    amounts = {el: max(1, round(v * scale)) for el, v in amounts_raw.items()}

    a = 4.0
    lattice = Lattice.cubic(a)
    positions = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ]
    species, coords = [], []
    idx = 0
    for el, amt in amounts.items():
        for _ in range(min(amt, 3)):  # cap at 3 per element
            if idx < len(positions):
                species.append(el)
                coords.append(positions[idx])
                idx += 1

    # Ensure at least 2 atoms
    while len(species) < 2:
        species.append('Li')
        coords.append(positions[idx % len(positions)])
        idx += 1

    return PmgStructure(lattice, species, coords)


class InvalidComposition(Exception):
    def __init__(self, message, suggestions=None):
        super().__init__(message)
        self.suggestions = suggestions or []

def lookup_or_build_structure(formula_str: str):
    """
    Returns (structure, material_id, source) where source is
    'dataset', 'synthetic', or raises ValueError with a message.
    Raises a dict-like ValueError with 'error', 'suggestions' keys via InvalidComposition.
    """
    # Parse composition
    try:
        comp = Composition(formula_str)
    except Exception:
        raise InvalidComposition(
            f"Cannot parse formula '{formula_str}'. "
            "Use standard chemical notation like LiCoO2.",
            suggestions=[]
        )

    elements = {str(el) for el in comp.elements}

    # Check elements are known
    if not elements.issubset(LI_ION_ELEMENTS):
        bad = elements - LI_ION_ELEMENTS
        suggestions = get_battery_suggestions(comp)
        raise InvalidComposition(
            f"'{comp.reduced_formula}' contains unknown/unsupported elements: {bad}.",
            suggestions=suggestions
        )

    # Full battery viability check
    valid, reason = is_viable_battery_material(comp)
    if not valid:
        suggestions = get_battery_suggestions(comp)
        raise InvalidComposition(
            f"'{comp.reduced_formula}' is not a valid battery material. {reason}",
            suggestions=suggestions
        )

    reduced = comp.reduced_formula.lower()

    # 1. Exact match
    if reduced in csv_index:
        mid      = csv_index[reduced]
        cif_path = f'Li_Battery_GNN_Dataset/cif/{mid}.cif'
        if os.path.exists(cif_path):
            return Structure.from_file(cif_path), mid, 'dataset'

    # 2. Same element set — pick closest composition by ratio similarity
    comp_elements = frozenset(str(el) for el in comp.elements)
    best_match, best_score = None, float('inf')
    for key, mid in csv_index.items():
        try:
            c2 = Composition(key)
            if frozenset(str(el) for el in c2.elements) == comp_elements:
                # Score = sum of squared ratio differences
                score = sum(
                    (comp.get_atomic_fraction(el) - c2.get_atomic_fraction(el)) ** 2
                    for el in comp.elements
                )
                if score < best_score:
                    cif_path = f'Li_Battery_GNN_Dataset/cif/{mid}.cif'
                    if os.path.exists(cif_path):
                        best_score = score
                        best_match = (mid, cif_path)
        except:
            continue
    if best_match:
        mid, cif_path = best_match
        return Structure.from_file(cif_path), mid, 'similar_composition'

    # 3. Subset match — same Li + TM elements, ignoring dopants like Al
    comp_tm = comp_elements & (ACTIVE_ELEMENTS | {'Li', 'O', 'P', 'F', 'S'})
    for key, mid in csv_index.items():
        try:
            c2 = Composition(key)
            c2_tm = frozenset(str(el) for el in c2.elements) & (ACTIVE_ELEMENTS | {'Li', 'O', 'P', 'F', 'S'})
            if comp_tm == c2_tm:
                cif_path = f'Li_Battery_GNN_Dataset/cif/{mid}.cif'
                if os.path.exists(cif_path):
                    return Structure.from_file(cif_path), mid, 'similar_composition'
        except:
            continue

    # 4. Synthetic fallback
    try:
        structure = build_synthetic_structure(comp)
        return structure, comp.reduced_formula, 'synthetic'
    except Exception:
        suggestions = get_battery_suggestions(comp)
        raise InvalidComposition(
            f"Could not build a structure for '{comp.reduced_formula}'.",
            suggestions=suggestions
        )



def cif_to_graph(cif_path):
    structure = Structure.from_file(cif_path)

    # Expand small structures into a supercell so the GNN has enough atoms/edges
    if len(structure) < 4:
        scale = max(2, int(np.ceil(4 / len(structure))))
        structure = structure * [scale, scale, scale]

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
            el.Z / 94.0, float(el.atomic_mass) / 238.0,
            eneg, el_row, el_group,
            site.coords[0] / 10.0, site.coords[1] / 10.0, site.coords[2] / 10.0,
            len(neighbors) / 12.0,
            1.0 if el.symbol == 'Li' else 0.0
        ])

    # Try increasing cutoffs until we get edges
    edge_index, edge_attr = [], []
    for cutoff in [4.5, 6.0, 8.0]:
        edge_index, edge_attr = [], []
        for i, site in enumerate(structure):
            try:
                for neighbor in structure.get_neighbors(site, cutoff):
                    j = neighbor.index  # use pymatgen's built-in index, no lookup needed
                    dist = neighbor.nn_distance
                    edge_index.append([i, j])
                    edge_attr.append([
                        dist / cutoff,
                        (1.0 / (dist + 1e-6)) / 2.0,
                        abs(site.specie.Z - neighbor.specie.Z) / 94.0
                    ])
            except Exception:
                continue
        if len(edge_index) > 0:
            break

    if len(edge_index) == 0:
        raise ValueError("No atomic bonds found — structure may be malformed")

    x          = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    batch      = torch.zeros(x.shape[0], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return data, structure

# ============================================================================
# DATABASE
# ============================================================================

DB_PATH = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        material_id TEXT,
        formula TEXT,
        voltage REAL,
        formation_energy REAL,
        density REAL,
        energy_above_hull REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        prediction_id INTEGER,
        rating INTEGER NOT NULL,
        comment TEXT,
        formula TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(prediction_id) REFERENCES predictions(id)
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        is_admin INTEGER NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    # Migrations for older DBs
    for col_sql in [
        'ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0',
        'ALTER TABLE users ADD COLUMN email TEXT',
        'ALTER TABLE feedback ADD COLUMN formula TEXT',
    ]:
        try:
            conn.execute(col_sql)
            conn.commit()
        except:
            pass
    # Seed default admin
    try:
        conn.execute(
            'INSERT INTO users (username, password_hash, is_admin) VALUES (?,?,1)',
            ('admin', hash_password('admin123'))
        )
        conn.commit()
        print("  ✅ Default admin created  →  admin / admin123")
    except:
        pass
    conn.commit()
    conn.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ============================================================================
# AUTH HELPERS
# ============================================================================

def _get_session(token):
    """Look up session from DB. Returns {'user_id', 'is_admin'} or None."""
    if not token:
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        'SELECT s.user_id, u.is_admin FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?',
        (token,)
    ).fetchone()
    conn.close()
    return {'user_id': row[0], 'is_admin': bool(row[1])} if row else None

def _save_session(token, user_id, is_admin):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT OR REPLACE INTO sessions (token, user_id, is_admin) VALUES (?,?,?)',
        (token, user_id, int(is_admin))
    )
    conn.commit()
    conn.close()

def _delete_session(token):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM sessions WHERE token=?', (token,))
    conn.commit()
    conn.close()

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        sess = _get_session(token)
        if not sess:
            return jsonify({'error': 'Unauthorized'}), 401
        request.user_id  = sess['user_id']
        request.is_admin = sess['is_admin']
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        sess = _get_session(token)
        if not sess:
            return jsonify({'error': 'Unauthorized'}), 401
        if not sess['is_admin']:
            return jsonify({'error': 'Admin access required'}), 403
        request.user_id  = sess['user_id']
        request.is_admin = True
        return f(*args, **kwargs)
    return decorated

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.json or {}
    email    = (data.get('email') or data.get('username') or '').strip().lower()
    password = data.get('password', '')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    conn = sqlite3.connect(DB_PATH)
    # Match by email column first, fall back to username for legacy accounts
    user = conn.execute(
        'SELECT id, username, is_admin FROM users WHERE (LOWER(email)=? OR LOWER(username)=?) AND password_hash=?',
        (email, email, hash_password(password))
    ).fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401

    token = hashlib.sha256(f"{email}{datetime.now().isoformat()}".encode()).hexdigest()
    _save_session(token, user[0], bool(user[2]))
    display_name = user[1] or email.split('@')[0]
    return jsonify({'token': token, 'username': display_name, 'is_admin': bool(user[2])})


@app.route('/api/register', methods=['POST'])
def register():
    data     = request.json or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password', '')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    if '@' not in email:
        return jsonify({'error': 'Invalid email address'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    username = email.split('@')[0]  # derive display name from email
    conn = sqlite3.connect(DB_PATH)
    # Check if email already registered
    existing = conn.execute('SELECT id FROM users WHERE LOWER(email)=?', (email,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'An account with this email already exists'}), 409
    try:
        conn.execute(
            'INSERT INTO users (username, email, password_hash, is_admin) VALUES (?,?,?,0)',
            (username, email, hash_password(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Email already registered'}), 409
    conn.close()
    return jsonify({'message': 'Account created successfully'})


@app.route('/api/me', methods=['GET'])
@require_auth
def me():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('SELECT username, is_admin FROM users WHERE id=?', (request.user_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'username': row[0], 'is_admin': bool(row[1])})


@app.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No CIF file uploaded'}), 400

    cif_file = request.files['file']
    if not cif_file.filename.endswith('.cif'):
        return jsonify({'error': 'File must be a .cif file'}), 400

    if not models:
        return jsonify({'error': 'Models not loaded'}), 503

    # Save temp file
    with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp:
        cif_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        graph, structure = cif_to_graph(tmp_path)
        graph = graph.to(device)

        v_preds, e_preds, d_preds, h_preds = [], [], [], []
        with torch.no_grad():
            for model in models:
                v, e, d, h = model(graph)
                v_preds.append(v.item())
                e_preds.append(e.item())
                d_preds.append(d.item())
                h_preds.append(h.item())

        v_raw = np.mean(v_preds)
        e_raw = np.mean(e_preds)
        d_raw = np.mean(d_preds)
        h_raw = np.mean(h_preds)

        # Inverse transform if scalers available
        if scalers:
            voltage          = float(scalers['voltage'].inverse_transform([[v_raw]])[0][0])
            formation_energy = float(scalers['energy'].inverse_transform([[e_raw]])[0][0])
            density          = float(scalers['density'].inverse_transform([[d_raw]])[0][0])
            energy_above_hull = float(scalers['hull'].inverse_transform([[h_raw]])[0][0])
        else:
            voltage, formation_energy, density, energy_above_hull = v_raw, e_raw, d_raw, h_raw

        formula     = structure.formula
        material_id = cif_file.filename.replace('.cif', '')

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            '''INSERT INTO predictions (user_id, material_id, formula, voltage,
               formation_energy, density, energy_above_hull)
               VALUES (?,?,?,?,?,?,?)''',
            (request.user_id, material_id, formula,
             voltage, formation_energy, density, energy_above_hull)
        )
        pred_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'prediction_id':  pred_id,
            'material_id': material_id,
            'formula': formula,
            'predictions': {
                'voltage':            round(voltage, 4),
                'formation_energy':   round(formation_energy, 4),
                'density':            round(density, 4),
                'energy_above_hull':  round(energy_above_hull, 4),
            },
            'units': {
                'voltage':            'V',
                'formation_energy':   'eV/atom',
                'density':            'g/cm³',
                'energy_above_hull':  'eV',
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/api/predict-composition', methods=['POST'])
@require_auth
def predict_composition():
    data    = request.json
    formula = (data.get('formula') or '').strip()
    if not formula:
        return jsonify({'error': 'Formula is required'}), 400

    if not models:
        return jsonify({'error': 'Models not loaded'}), 503

    try:
        structure, material_id, source = lookup_or_build_structure(formula)
    except InvalidComposition as e:
        return jsonify({'error': str(e), 'compatible': False, 'suggestions': e.suggestions}), 422
    except ValueError as e:
        return jsonify({'error': str(e), 'compatible': False, 'suggestions': []}), 422

    # Write to temp CIF and run prediction
    with tempfile.NamedTemporaryFile(suffix='.cif', delete=False, mode='w') as tmp:
        structure.to(fmt='cif', filename=tmp.name)
        tmp_path = tmp.name

    try:
        graph, _ = cif_to_graph(tmp_path)
        result   = run_inference(graph)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            '''INSERT INTO predictions (user_id, material_id, formula, voltage,
               formation_energy, density, energy_above_hull)
               VALUES (?,?,?,?,?,?,?)''',
            (request.user_id, str(material_id), structure.formula,
             result['predictions']['voltage'], result['predictions']['formation_energy'],
             result['predictions']['density'],  result['predictions']['energy_above_hull'])
        )
        pred_id = cursor.lastrowid
        conn.commit()
        conn.close()

        source_msg = {
            'dataset':             '✅ Found in dataset',
            'similar_composition': '⚠️ Similar composition used (exact not found)',
            'synthetic':           '⚠️ Synthetic structure generated (not in dataset)',
        }.get(source, source)

        return jsonify({
            'compatible':     True,
            'prediction_id':  pred_id,
            'material_id':    str(material_id),
            'formula':        structure.formula,
            'source':         source,
            'source_message': source_msg,
            **result
        })

    except Exception as e:
        return jsonify({'error': str(e), 'compatible': False}), 500
    finally:
        os.unlink(tmp_path)



def run_inference(graph):
    """Run all 3 models, return mean + std for each property (in original scale)."""
    graph = graph.to(device)
    v_p, e_p, d_p, h_p = [], [], [], []
    with torch.no_grad():
        for m in models:
            v, e, d, h = m(graph)
            v_p.append(v.item()); e_p.append(e.item())
            d_p.append(d.item()); h_p.append(h.item())

    def inv(sc, vals):
        mean = float(sc.inverse_transform([[np.mean(vals)]])[0][0]) if scalers else np.mean(vals)
        # std in original space ≈ std_scaled * scale_
        std  = float(np.std(vals) * sc.scale_[0]) if scalers else float(np.std(vals))
        return round(mean, 4), round(std, 4)

    v_mean, v_std = inv(scalers.get('voltage'), v_p) if scalers else (round(np.mean(v_p),4), round(float(np.std(v_p)),4))
    e_mean, e_std = inv(scalers.get('energy'),  e_p) if scalers else (round(np.mean(e_p),4), round(float(np.std(e_p)),4))
    d_mean, d_std = inv(scalers.get('density'), d_p) if scalers else (round(np.mean(d_p),4), round(float(np.std(d_p)),4))
    h_mean, h_std = inv(scalers.get('hull'),    h_p) if scalers else (round(np.mean(h_p),4), round(float(np.std(h_p)),4))

    return {
        'predictions': {
            'voltage':           v_mean, 'formation_energy': e_mean,
            'density':           d_mean, 'energy_above_hull': h_mean,
        },
        'confidence': {
            'voltage':           v_std,  'formation_energy': e_std,
            'density':           d_std,  'energy_above_hull': h_std,
        },
        'units': {'voltage':'V','formation_energy':'eV/atom','density':'g/cm³','energy_above_hull':'eV'}
    }


@app.route('/api/predict-compare', methods=['POST'])
@require_auth
def predict_compare():
    """Compare two formulas side by side."""
    data = request.json or {}
    # Accept either {formula_a, formula_b} or {formulas: [...]}
    formula_a = data.get('formula_a') or (data.get('formulas', [None, None])[0])
    formula_b = data.get('formula_b') or (data.get('formulas', [None, None])[1])
    if not formula_a or not formula_b:
        return jsonify({'error': 'Provide formula_a and formula_b'}), 400
    if not models:
        return jsonify({'error': 'Models not loaded'}), 503

    def run_for(formula):
        structure, material_id, source = lookup_or_build_structure(formula.strip())
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False, mode='w') as tmp:
            structure.to(fmt='cif', filename=tmp.name)
            tmp_path = tmp.name
        try:
            graph, _ = cif_to_graph(tmp_path)
            result = run_inference(graph)
            result['formula']     = structure.formula
            result['material_id'] = str(material_id)
            result['source']      = source
            return result
        finally:
            os.unlink(tmp_path)

    try:
        ra = run_for(formula_a)
    except InvalidComposition as e:
        return jsonify({'error': str(e), 'suggestions': e.suggestions, 'field': 'a'}), 422
    except Exception as e:
        return jsonify({'error': f'Formula A error: {e}'}), 422
    try:
        rb = run_for(formula_b)
    except InvalidComposition as e:
        return jsonify({'error': str(e), 'suggestions': e.suggestions, 'field': 'b'}), 422
    except Exception as e:
        return jsonify({'error': f'Formula B error: {e}'}), 422

    return jsonify({'result_a': ra, 'result_b': rb})


@app.route('/api/predict-batch', methods=['POST'])
@require_auth
def predict_batch():
    """Predict a list of formulas (max 20)."""
    data     = request.json or {}
    formulas = data.get('formulas', [])
    if not formulas:
        return jsonify({'error': 'No formulas provided'}), 400
    if len(formulas) > 20:
        return jsonify({'error': 'Maximum 20 formulas per batch'}), 400
    if not models:
        return jsonify({'error': 'Models not loaded'}), 503

    results = []
    for formula in formulas:
        formula = str(formula).strip()
        try:
            structure, material_id, source = lookup_or_build_structure(formula)
            with tempfile.NamedTemporaryFile(suffix='.cif', delete=False, mode='w') as tmp:
                structure.to(fmt='cif', filename=tmp.name)
                tmp_path = tmp.name
            try:
                graph, _ = cif_to_graph(tmp_path)
                result   = run_inference(graph)
                result['formula']     = structure.formula
                result['material_id'] = str(material_id)
                result['source']      = source
                result['status']      = 'ok'
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    '''INSERT INTO predictions (user_id, material_id, formula, voltage,
                       formation_energy, density, energy_above_hull) VALUES (?,?,?,?,?,?,?)''',
                    (request.user_id, str(material_id), structure.formula,
                     result['predictions']['voltage'], result['predictions']['formation_energy'],
                     result['predictions']['density'],  result['predictions']['energy_above_hull'])
                )
                conn.commit(); conn.close()
                results.append(result)
            except Exception as e:
                results.append({'formula': formula, 'status': 'error', 'error': str(e)})
            finally:
                try: os.unlink(tmp_path)
                except: pass
        except Exception as e:
            results.append({'formula': formula, 'status': 'error', 'error': str(e)})

    return jsonify({'results': results})


@app.route('/api/history', methods=['GET'])
@require_auth
def history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        '''SELECT material_id, formula, voltage, formation_energy,
                  density, energy_above_hull, created_at
           FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 50''',
        (request.user_id,)
    ).fetchall()
    conn.close()

    return jsonify([{
        'material_id':        r[0],
        'formula':            r[1],
        'voltage':            r[2],
        'formation_energy':   r[3],
        'density':            r[4],
        'energy_above_hull':  r[5],
        'created_at':         r[6],
    } for r in rows])


@app.route('/api/logout', methods=['POST'])
@require_auth
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    _delete_session(token)
    return jsonify({'message': 'Logged out'})


# ============================================================================
# ADMIN ROUTES
# ============================================================================

@app.route('/api/feedback', methods=['POST'])
@require_auth
def submit_feedback():
    data    = request.json or {}
    rating  = data.get('rating')
    comment = data.get('comment', '').strip()
    pred_id = data.get('prediction_id')
    formula = data.get('formula', '').strip()
    if not rating or not (1 <= int(rating) <= 5):
        return jsonify({'error': 'Rating must be 1-5'}), 400
    conn = sqlite3.connect(DB_PATH)
    # If no formula, try to get it from the prediction
    if not formula and pred_id:
        row = conn.execute('SELECT formula FROM predictions WHERE id=?', (pred_id,)).fetchone()
        if row:
            formula = row[0]
    # If no prediction_id but formula given, look up the latest prediction for this user+formula
    if not pred_id and formula:
        row = conn.execute(
            'SELECT id FROM predictions WHERE user_id=? AND formula=? ORDER BY created_at DESC LIMIT 1',
            (request.user_id, formula)
        ).fetchone()
        if row:
            pred_id = row[0]
    conn.execute(
        'INSERT INTO feedback (user_id, prediction_id, rating, comment, formula) VALUES (?,?,?,?,?)',
        (request.user_id, pred_id, int(rating), comment, formula)
    )
    conn.commit()
    conn.close()
    return jsonify({'message': 'Feedback saved'})


@app.route('/api/admin/stats', methods=['GET'])
@require_admin
def admin_stats():
    conn = sqlite3.connect(DB_PATH)
    total_users  = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_preds  = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    today        = datetime.now().strftime('%Y-%m-%d')
    preds_today  = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE created_at LIKE ?", (today+'%',)
    ).fetchone()[0]

    # Average rating
    avg_row = conn.execute('SELECT AVG(rating) FROM feedback').fetchone()
    avg_rating = round(avg_row[0], 2) if avg_row[0] else None

    # Rating distribution [1..5]
    rating_dist = []
    for i in range(1, 6):
        c = conn.execute('SELECT COUNT(*) FROM feedback WHERE rating=?', (i,)).fetchone()[0]
        rating_dist.append(c)

    # Daily prediction counts for last 7 days
    daily_labels, daily_counts = [], []
    for i in range(6, -1, -1):
        from datetime import timedelta
        day = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        cnt = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE created_at LIKE ?", (day+'%',)
        ).fetchone()[0]
        daily_labels.append(day[5:])  # MM-DD
        daily_counts.append(cnt)

    active_sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
    conn.close()
    return jsonify({
        'total_users':       total_users,
        'total_predictions': total_preds,
        'predictions_today': preds_today,
        'active_sessions':   active_sessions,
        'avg_rating':        avg_rating,
        'rating_dist':       rating_dist,
        'daily_labels':      daily_labels,
        'daily_counts':      daily_counts,
    })


@app.route('/api/admin/users', methods=['GET'])
@require_admin
def admin_users():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT u.id, u.username, u.is_admin, u.created_at,
               COUNT(DISTINCT p.id) as pred_count,
               ROUND(AVG(f.rating), 1) as avg_rating
        FROM users u
        LEFT JOIN predictions p ON p.user_id = u.id
        LEFT JOIN feedback f ON f.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    ''').fetchall()
    conn.close()
    return jsonify([{
        'id':         r[0],
        'username':   r[1],
        'is_admin':   bool(r[2]),
        'created_at': r[3],
        'pred_count': r[4],
        'avg_rating': r[5],
    } for r in rows])


@app.route('/api/admin/feedback', methods=['GET'])
@require_admin
def admin_feedback():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT f.id, u.username, COALESCE(f.formula, p.formula, '—') as formula,
               f.rating, f.comment, f.created_at
        FROM feedback f
        JOIN users u ON u.id = f.user_id
        LEFT JOIN predictions p ON p.id = f.prediction_id
        ORDER BY f.created_at DESC
        LIMIT 100
    ''').fetchall()
    conn.close()
    return jsonify([{
        'id':         r[0],
        'username':   r[1],
        'formula':    r[2],
        'rating':     r[3],
        'comment':    r[4],
        'created_at': r[5],
    } for r in rows])


@app.route('/api/admin/users/<int:uid>', methods=['DELETE'])
@require_admin
def admin_delete_user(uid):
    if uid == request.user_id:
        return jsonify({'error': 'Cannot delete yourself'}), 400
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM predictions WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM predictions WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM sessions WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM users WHERE id=?', (uid,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'User deleted'})


@app.route('/api/admin/users/<int:uid>/toggle-admin', methods=['PATCH'])
@require_admin
def admin_toggle_admin(uid):
    if uid == request.user_id:
        return jsonify({'error': 'Cannot change your own admin status'}), 400
    conn = sqlite3.connect(DB_PATH)
    current = conn.execute('SELECT is_admin FROM users WHERE id=?', (uid,)).fetchone()
    if not current:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    new_val = 0 if current[0] else 1
    conn.execute('UPDATE users SET is_admin=? WHERE id=?', (new_val, uid))
    conn.commit()
    conn.close()
    return jsonify({'is_admin': bool(new_val)})


@app.route('/api/admin/users/<int:uid>/reset-password', methods=['POST'])
@require_admin
def admin_reset_password(uid):
    new_pw = (request.json or {}).get('password', '').strip()
    if len(new_pw) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE users SET password_hash=? WHERE id=?', (hash_password(new_pw), uid))
    conn.execute('DELETE FROM sessions WHERE user_id=?', (uid,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Password reset'})


@app.route('/api/admin/predictions', methods=['GET'])
@require_admin
def admin_predictions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT p.id, u.username, p.formula, p.material_id,
               p.voltage, p.formation_energy, p.density, p.energy_above_hull, p.created_at
        FROM predictions p
        JOIN users u ON u.id = p.user_id
        ORDER BY p.created_at DESC
        LIMIT 200
    ''').fetchall()
    conn.close()
    return jsonify([{
        'id':                r[0],
        'username':          r[1],
        'formula':           r[2],
        'material_id':       r[3],
        'voltage':           r[4],
        'formation_energy':  r[5],
        'density':           r[6],
        'energy_above_hull': r[7],
        'created_at':        r[8],
    } for r in rows])


@app.route('/api/admin/predictions/<int:pid>', methods=['DELETE'])
@require_admin
def admin_delete_prediction(pid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM predictions WHERE id=?', (pid,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Prediction deleted'})


# Serve frontend
@app.route('/', defaults={'path': ''})
@app.route('/ui')
@app.route('/<path:path>')
def serve(path=''):
    # Don't intercept API routes
    if str(path).startswith('api/'):
        from flask import abort
        abort(404)
    static = app.static_folder
    if path and os.path.exists(os.path.join(static, path)):
        return send_from_directory(static, path)
    index = os.path.join(static, 'index.html')
    if os.path.exists(index):
        return send_from_directory(static, 'index.html')
    return jsonify({'error': 'Frontend not found'}), 404


if __name__ == '__main__':
    init_db()
    load_models()
    load_csv_index()
    print("\n🚀 Server running at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
