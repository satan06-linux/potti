"""
🚀 Start the Battery Predictor Web App
Run: python run.py
Then open: http://localhost:5000/ui
"""
import os, sys

# Ensure we run from the project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import app, init_db, load_models, load_csv_index

if __name__ == '__main__':
    init_db()
    load_models()
    load_csv_index()
    print("\n" + "="*50)
    print("🔋 Battery Predictor Web App")
    print("="*50)
    print("🌐 Open: http://localhost:5000/ui")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
