#!/usr/bin/env python
import os
import time
import argparse
import requests
import pandas as pd
from tensorflow.keras.models import load_model

# ── Configuration ────────────────────────────────────────────────────────────────
DEFAULT_AGG_URL = "http://localhost:5000"
NUM_FEATURES    = 41

# ── CLI ───────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int,   required=True)
parser.add_argument("--data_file", type=str,   required=True)
parser.add_argument("--agg_url", type=str,     default=DEFAULT_AGG_URL)
args = parser.parse_args()

CID       = args.client_id
DATA_FILE = args.data_file
AGG_URL   = args.agg_url.rstrip("/")
MODEL_F   = f"downloaded_model_{CID}.h5"

def upload_data():
    print(f"[DEV {CID}] Uploading {DATA_FILE}")
    with open(DATA_FILE, "rb") as fp:
        r = requests.post(
            f"{AGG_URL}/upload-data",
            files={"file": (os.path.basename(DATA_FILE), fp)},
            data={"client_id": CID},
        )
    print(f"[DEV {CID}] Upload response: {r.status_code} {r.text}")
    r.raise_for_status()

def wait_for_training():
    print(f"[DEV {CID}] Waiting for aggregator to finish training...")
    while True:
        r = requests.get(f"{AGG_URL}/status")
        st = r.json().get("status")
        print(f"[DEV {CID}] Status: {st}")
        if st == "trained":
            break
        time.sleep(1)

def download_and_evaluate():
    print(f"[DEV {CID}] Downloading model")
    r = requests.get(f"{AGG_URL}/model", stream=True)
    r.raise_for_status()
    with open(MODEL_F, "wb") as out:
        for chunk in r.iter_content(1024):
            out.write(chunk)
    print(f"[DEV {CID}] Model saved to {MODEL_F}")

    df = pd.read_csv(DATA_FILE)
    X  = df.iloc[:, :NUM_FEATURES].values.astype("float32")
    y  = df.iloc[:, NUM_FEATURES].values.astype("int32")

    model = load_model(MODEL_F)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"[DEV {CID}] Local eval → loss: {loss:.4f}, acc: {acc:.4f}")

if __name__ == "__main__":
    # 1) upload raw data
    upload_data()
    # 2) wait until aggregator finishes (includes 5s pre-wait + training + 5s post-wait)
    wait_for_training()
    # 3) download and run local inference automatically
    download_and_evaluate()
