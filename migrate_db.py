"""Run this to migrate DB to latest schema."""
import sqlite3, hashlib

conn = sqlite3.connect('predictions.db')

# Add sessions table
conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    is_admin INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
)''')

# Add feedback table
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

for col_sql in [
    'ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0',
    'ALTER TABLE users ADD COLUMN email TEXT',
    'ALTER TABLE feedback ADD COLUMN formula TEXT',
]:
    try:
        conn.execute(col_sql)
        conn.commit()
        print('Applied:', col_sql[:50])
    except Exception as e:
        print('Skip:', str(e)[:60])

def h(pw): return hashlib.sha256(pw.encode()).hexdigest()

# Ensure admin user exists with email
admin = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()
if admin:
    conn.execute("UPDATE users SET is_admin=1, email='admin@batterygnn.com' WHERE username='admin'")
    conn.commit()
    print('Updated admin email → admin@batterygnn.com / admin123')
else:
    conn.execute(
        'INSERT INTO users (username, email, password_hash, is_admin) VALUES (?,?,?,1)',
        ('admin', 'admin@batterygnn.com', h('admin123'))
    )
    conn.commit()
    print('Created admin → admin@batterygnn.com / admin123')

conn.commit()
conn.close()

conn = sqlite3.connect('predictions.db')
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('\nTables:', tables)
conn.close()
print('\nDone. Restart Flask: python app.py')
