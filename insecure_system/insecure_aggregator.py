#!/usr/bin/env python
import os
import time
import threading
import pandas as pd
from flask import Flask, request, jsonify, send_file
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────────
NUM_FEATURES   = 41
NUM_CLASSES    = 5
NUM_DEVICES    = 3
UPLOAD_DIR     = "uploads"
MODEL_FILE     = "global_model.h5"
PLOT_DIR       = "plots"
ROUNDS         = 5

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Status management ────────────────────────────────────────────────────────────
status_lock    = threading.Lock()
_status        = "waiting_uploads"   # waiting_uploads → pending_start → training → trained
thread_started = False

def set_status(s: str):
    global _status
    with status_lock:
        _status = s

def get_status() -> str:
    with status_lock:
        return _status

# ── Model builder ────────────────────────────────────────────────────────────────
def build_model():
    m = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(NUM_FEATURES,)),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    m.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )
    return m

# ── Training routine (background thread) ────────────────────────────────────────
def training_routine(upload_paths):
    # 1) Pre-start delay
    time.sleep(5)
    set_status("training")
    print("[AGG] Starting training")

    # 2) Load & concatenate all device uploads
    dfs = [pd.read_csv(p) for p in upload_paths]
    df_all = pd.concat(dfs, ignore_index=True)
    X_all = df_all.iloc[:, :NUM_FEATURES].values.astype("float32")
    y_all = df_all.iloc[:, NUM_FEATURES].values.astype("int32")

    # 3) Prepare per-device datasets
    dev_sets = []
    for i in range(NUM_DEVICES):
        df = pd.read_csv(os.path.join(UPLOAD_DIR, f"device_{i}.csv"))
        X = df.iloc[:, :NUM_FEATURES].values.astype("float32")
        y = df.iloc[:, NUM_FEATURES].values.astype("int32")
        dev_sets.append((X, y))

    # 4) Training loop
    model      = build_model()
    agg_losses = []
    agg_accs   = []
    dev_losses = [[] for _ in range(NUM_DEVICES)]
    dev_accs   = [[] for _ in range(NUM_DEVICES)]

    for r in range(1, ROUNDS + 1):
        model.fit(X_all, y_all, epochs=1, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_all, y_all, verbose=0)
        agg_losses.append(loss)
        agg_accs.append(acc)
        print(f"[Round {r}] AGG → loss={loss:.4f}, acc={acc:.4f}")

        for i, (X_dev, y_dev) in enumerate(dev_sets):
            ldev, adev = model.evaluate(X_dev, y_dev, verbose=0)
            dev_losses[i].append(ldev)
            dev_accs[i].append(adev)
            print(f"[Round {r}] Device {i} → loss={ldev:.4f}, acc={adev:.4f}")

    # ----- New: Evaluate on held-out test set aggregatorTest.csv -----
    try:
        df_test = pd.read_csv("aggregatorTest.csv")
        X_test = df_test.iloc[:, :NUM_FEATURES].values.astype("float32")
        y_test = df_test.iloc[:, NUM_FEATURES].values.astype("int32")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"[AGG Test] loss={test_loss:.4f}, acc={test_acc:.4f}")
    except FileNotFoundError:
        print("[AGG Test] aggregatorTest.csv not found, skipping held-out evaluation")
    # --------------------------------------------------------------------

    # 5) Save the final global model
    model.save(MODEL_FILE)
    print(f"[AGG] Final model saved to {MODEL_FILE}")

    # 6) Summary
    print("\n[AGG] Full training summary:")
    for r in range(ROUNDS):
        line = f" Round {r+1}: AGG loss={agg_losses[r]:.4f}, acc={agg_accs[r]:.4f};"
        for i in range(NUM_DEVICES):
            line += f" Dev{i} loss={dev_losses[i][r]:.4f}, acc={dev_accs[i][r]:.4f};"
        print(line)
    print(f"\n[AGG] Final Aggregator → loss={agg_losses[-1]:.4f}, acc={agg_accs[-1]:.4f}")

    # 7) Post-training delay
    time.sleep(5)
    set_status("trained")

    # 8) Plot 1: Aggregator loss per round
    rounds = list(range(1, ROUNDS + 1))
    plt.figure()
    plt.plot(rounds, agg_losses, marker="o")
    plt.title("Aggregator Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(PLOT_DIR, "insecure_loss.png"))
    plt.close()

    # 9) Plot 2: Aggregator accuracy per round
    plt.figure()
    plt.plot(rounds, agg_accs, marker="o")
    plt.title("Aggregator Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(PLOT_DIR, "insecure_accuracy.png"))
    plt.close()

    # 10) Plot 3: All devices' validation loss curves
    plt.figure()
    for i in range(NUM_DEVICES):
        plt.plot(rounds, dev_losses[i], marker="o", label=f"Device {i}")
    plt.title("Device Validation Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "insecure_client_loss_rounds.png"))
    plt.close()


# ── Flask app ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/upload-data", methods=["POST"])
def upload_data():
    global thread_started
    cid = request.form.get("client_id")
    if cid is None:
        return "Missing client_id", 400
    f = request.files.get("file")
    if not f:
        return "Missing file", 400

    dest = os.path.join(UPLOAD_DIR, f"device_{cid}.csv")
    f.save(dest)
    print(f"[AGG] Received upload from device {cid}")

    # Once all uploads are present, start training once
    present = {fn for fn in os.listdir(UPLOAD_DIR) if fn.endswith(".csv")}
    if len(present) == NUM_DEVICES and not thread_started:
        thread_started = True
        set_status("pending_start")
        upload_paths = [os.path.join(UPLOAD_DIR, f"device_{i}.csv") for i in range(NUM_DEVICES)]
        threading.Thread(target=training_routine, args=(upload_paths,), daemon=True).start()

    return jsonify(status="ok")

@app.route("/status", methods=["GET"])
def status_endpoint():
    return jsonify(status=get_status())

@app.route("/model", methods=["GET"])
def serve_model():
    if not os.path.exists(MODEL_FILE):
        return "Model not found", 404
    return send_file(MODEL_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
