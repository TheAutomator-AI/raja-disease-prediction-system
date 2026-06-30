import os
import json
import joblib
import numpy as np
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- BULLETPROOF PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PARENT_DIR, 'models')

DISEASES = ['heart', 'diabetes', 'breast']
loaded_models = {}
loaded_scalers = {}
loaded_metrics = {}

# Load Models on Startup
# ... (Keep the rest of your app.py exactly the same from here down)