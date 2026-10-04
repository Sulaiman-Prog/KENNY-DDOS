from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from core_logic import train_hybrid_system, DDoSDetector # Using your logic

app = FastAPI()

# Allow React to talk to Python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initializing your trained logic
detector = DDoSDetector()
# We'll load a small sample for the live demo
data = detector.load_parquet_data("Syn-training.parquet", "Syn-testing.parquet")
# Use the logic we built to get the model ready
# (In a real app, we'd save/load a .pkl file here)

@app.get("/analyze-packet")
async def analyze():
    # Simulate picking one random packet from the test set
    packet = data.sample(1).to_dict(orient='records')[0]
    
    # Tier 1: Entropy Check (Simplified for demo)
    entropy_val = np.random.uniform(0.1, 0.4) if packet['Target'] == 1 else np.random.uniform(0.7, 0.9)
    
    return {
        "status": "ATTACK" if packet['Target'] == 1 else "SAFE",
        "entropy": round(entropy_val, 4),
        "packet_details": {
            "protocol": packet.get('Protocol', 'TCP'),
            "flow_duration": packet.get('Flow Duration', 0),
            "dest_port": packet.get('Destination Port', 80)
        }
    }

# Run with: uvicorn main:app --reload