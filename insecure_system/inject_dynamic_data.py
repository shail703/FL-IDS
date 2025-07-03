import argparse
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import random

NUM_FEATURES = 41
TOTAL_SAMPLES = 100
LABEL_INDEX = {
    0: "normal",
    1: "dos",
    2: "probe",
    3: "r2l",
    4: "u2r"
}

def generate_unlabeled_row(df_features_only):
    row = df_features_only.sample(n=1, replace=True).iloc[0]
    return row.values

def load_model_latest(model_path="global_model.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")
    return load_model(model_path)

def process_predictions(X, model, conf_hi, conf_lo, allow_low_conf):
    probs = model.predict(X)
    confidences = np.max(probs, axis=1)
    labels = np.argmax(probs, axis=1)

    accepted, rejected = [], []
    for i in range(len(X)):
        row = list(X[i])
        label = int(labels[i])
        conf = confidences[i]
        if conf >= conf_hi:
            accepted.append(row + [label, "model_high_conf"])
        elif allow_low_conf and conf >= conf_lo:
            accepted.append(row + [label, "model_low_conf"])
        else:
            rejected.append(row + [label, conf])
    return accepted, rejected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_file", required=True, help="Target device file (e.g., device1.csv)")
    parser.add_argument("--confidence_threshold_high", type=float, default=0.9)
    parser.add_argument("--confidence_threshold_low", type=float, default=0.7)
    parser.add_argument("--allow_low_conf", action="store_true")
    args = parser.parse_args()

    device_filename = os.path.basename(args.device_file)
    device_id = device_filename.replace("device", "").replace(".csv", "")
    rejected_file = f"rejected_predictions_device{device_id}.csv"

    print(f"Using {args.device_file} as source and target...")
    base_df = pd.read_csv(args.device_file)
    if base_df.shape[1] != NUM_FEATURES + 1:
        raise ValueError("Device CSV must have 41 features + 1 label.")

    df_features = base_df.drop(columns=["attack_label"])
    model = load_model_latest()

    accepted_all = []
    rejected_all = []

    # === STEP 1: Reprocess rejected samples if available ===
    if os.path.exists(rejected_file):
        print(f"Re-evaluating rejected samples for device {device_id}...")
        df_rejected = pd.read_csv(rejected_file)
        if df_rejected.shape[1] != NUM_FEATURES + 2:
            raise ValueError(f"{rejected_file} must have 41 features + predicted_label + confidence")

        X_rej = df_rejected.iloc[:, :NUM_FEATURES].values
        re_accepted, re_remaining = process_predictions(
            X_rej, model, args.confidence_threshold_high, args.confidence_threshold_low, args.allow_low_conf
        )
        accepted_all.extend(re_accepted)
        rejected_all.extend(re_remaining)

    # === STEP 2: Generate new synthetic samples ===
    print("Generating and classifying 100 new synthetic samples...")
    X_new = np.array([generate_unlabeled_row(df_features) for _ in range(TOTAL_SAMPLES)])
    new_accepted, new_rejected = process_predictions(
        X_new, model, args.confidence_threshold_high, args.confidence_threshold_low, args.allow_low_conf
    )
    accepted_all.extend(new_accepted)
    rejected_all.extend(new_rejected)

    # === Save accepted data (strip label_source before saving) ===
    if accepted_all:
        df_accepted = pd.DataFrame(accepted_all,
            columns=[f"f{i}" for i in range(NUM_FEATURES)] + ["attack_label", "label_source"]
        )
        df_accepted.drop(columns=["label_source"]).to_csv(args.device_file, mode="a", header=False, index=False)
        print(f"Appended {len(df_accepted)} rows to {args.device_file}")

    # === Save rejected data for this device ===
    if rejected_all:
        df_rejected_out = pd.DataFrame(rejected_all)
        df_rejected_out.columns = [f"f{i}" for i in range(NUM_FEATURES)] + ["predicted_label", "confidence"]
        df_rejected_out.to_csv(rejected_file, index=False)
        print(f"{len(df_rejected_out)} low-confidence rows retained in {rejected_file}")
    else:
        if os.path.exists(rejected_file):
            os.remove(rejected_file)
        print(f"No low-confidence samples remain. Cleaned {rejected_file}.")

    # === Print summary ===
    print("\nFinal Accepted Class Distribution:")
    accepted_labels = [int(row[NUM_FEATURES]) for row in accepted_all]
    counts = pd.Series(accepted_labels).value_counts().sort_index()
    for cls in range(5):
        print(f"{LABEL_INDEX[cls]} ({cls}): {counts.get(cls, 0)}")

if __name__ == "__main__":
    main()
