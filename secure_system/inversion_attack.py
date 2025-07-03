#!/usr/bin/env python
"""
inversion_attack.py
───────────────────
Compare 10 actual rows vs. their inverted feature vectors (41 features each),
and export the results to a single Excel file with two sheets:

  • “interleaved” — Actual and Recovered rows interleaved
  • “errors”      — Per-row L2 reconstruction error

Columns are now numbered 1–41 instead of 0–40.
"""
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

NUM_FEATURES = 41
NUM_CLASSES  = 5

def build_model():
    inp = tf.keras.Input(shape=(NUM_FEATURES,), dtype=tf.float32)
    x   = tf.keras.layers.Dense(64, activation="relu")(inp)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out)

def load_grad(path):
    return np.load(path)

def flatten_grads(grads):
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def inversion_attack(captured_grad, target_label=0, steps=10, lr=0.1):
    model = build_model()
    x_var = tf.Variable(tf.random.normal([1, NUM_FEATURES]), trainable=True)
    y_true = tf.one_hot([target_label], depth=NUM_CLASSES)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    cap = tf.constant(captured_grad, dtype=tf.float32)

    for _ in range(steps):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_var)
            preds = model(x_var, training=False)
            loss_m = tf.keras.losses.categorical_crossentropy(y_true, preds)
            grads  = tape.gradient(loss_m, model.trainable_weights)
            flat   = flatten_grads(grads)
            match  = tf.reduce_sum((flat - cap) ** 2)
            reg    = tf.reduce_sum((x_var / 10.0) ** 2)
            total  = match + 0.1 * reg
        grad_x = tape.gradient(total, x_var)
        del tape
        opt.apply_gradients([(grad_x, x_var)])

    return x_var.numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_file", type=str, required=True,
                        help="Path to the captured gradient .npy file")
    parser.add_argument("--dp",        action="store_true",
                        help="Flag if the gradient is DP-noisy")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the device CSV (41 feature columns)")
    args = parser.parse_args()

    print(f"[+] Loading gradient: {args.grad_file} ({'DP-noisy' if args.dp else 'raw'})")
    grad = load_grad(args.grad_file)

    # 1) Read first 10 actual rows (features only)
    df       = pd.read_csv(args.data_file)
    actual10 = df.iloc[:10, :NUM_FEATURES].reset_index(drop=True)

    # Rename columns from 1 to 41
    numbered_cols = [str(i) for i in range(1, NUM_FEATURES+1)]
    actual10.columns = numbered_cols

    # 2) Invert once (10 GD steps)
    print("[+] Running inversion (10 steps)…")
    recovered = inversion_attack(grad, target_label=0, steps=10)

    # 3) Build DataFrame of recovered repeated 10× and rename columns
    rec10 = pd.DataFrame(
        np.tile(recovered, (10, 1)),
        columns=numbered_cols
    )

    # 4) Build interleaved DataFrame: Actual/Recovered pairs
    interleaved_rows = []
    for i in range(10):
        act = actual10.iloc[i].copy()
        rec = rec10.iloc[i].copy()
        act["Type"] = "Actual"
        rec["Type"] = "Recovered"
        interleaved_rows.append(act)
        interleaved_rows.append(rec)
    interleaved_df = pd.DataFrame(interleaved_rows).reset_index(drop=True)

    # 5) Compute per-row L2 errors
    errors = np.linalg.norm(actual10.values - rec10.values, axis=1)
    err_df = pd.DataFrame({
        "Row (1–10)": np.arange(1, 11),
        "L2_error":   np.round(errors, 4)
    })

    # 6) Export to a single Excel file with two sheets
    out_file = "inversion_results.xlsx"
    with pd.ExcelWriter(out_file) as writer:
        interleaved_df.to_excel(writer, sheet_name="interleaved", index=False)
        err_df.to_excel(writer, sheet_name="errors",      index=False)
    print(f"[+] Exported interleaved & errors to {out_file}")

    print("\n" + ("[!] DP-noisy gradient → expect large errors."
                  if args.dp else "[!] Raw gradient → expect small errors."))
