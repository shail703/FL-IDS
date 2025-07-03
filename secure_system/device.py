#!/usr/bin/env python
"""
device.py
---------
Flower 1.4.0-compatible client code for NSL-KDD, updated to dump
each round’s weight-delta vector (with or without DP noise).

 - Loads a local CSV (e.g., device0.csv) for training/testing.
 - Uses adversarial training + DP-SGD with an adaptive noise schedule.
 - Saves the local model after each FL round (HDF5 format, no optimizer).
 - Dumps raw or noisy weight deltas as .npy for inversion attack.
 - Communicates with the aggregator using the Flower 1.4.0 Parameters API.
 - The aggregator sets round_number in fit_ins.config; we read it here.
 - Devices start immediately (no startup delay).
"""

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import os
from typing import List, Tuple, Dict, Optional
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from phe import paillier
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters

# Constants
NUM_FEATURES = 41
NUM_CLASSES = 5

# Paillier keypair (dummy) for compatibility with aggregator demo
np.random.seed(999)
dummy_pubkey, _ = paillier.generate_paillier_keypair(n_length=1024)
public_key = paillier.PaillierPublicKey(dummy_pubkey.n)

# Ensure model dump directory exists
os.makedirs("client_models", exist_ok=True)

# === ADDED: helper to flatten a list of weight arrays into one vector ===
def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights], axis=0)
# ========================================================================


class AdversarialModel(keras.Model):
    def train_step(self, data):
        x, y = data
        # Regular gradient
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)

        # Adversarial gradient
        with tf.GradientTape() as tape_adv:
            tape_adv.watch(x)
            y_pred_adv = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred_adv, regularization_losses=self.losses)
        grad_x = tape_adv.gradient(loss_adv, x)
        epsilon = 0.01
        x_adv = x + epsilon * tf.sign(grad_x)
        with tf.GradientTape() as tape2:
            y_pred_adv2 = self(x_adv, training=True)
            loss_adv2 = self.compiled_loss(y, y_pred_adv2, regularization_losses=self.losses)
        grads_adv = tape2.gradient(loss_adv2, self.trainable_variables)

        # Combine gradients
        combined_grads = [(g + ga) / 2 for g, ga in zip(grads, grads_adv)]
        # Mark DP gradient usage
        self.optimizer._was_dp_gradients_called = True
        self.optimizer.apply_gradients(zip(combined_grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# builds a simple neural network:
# Input layer: 41 features
# Hidden layer: 64 neurons (ReLU activation)
# Output layer: 5 neurons (softmax activation for classification)


def create_adversarial_model(num_features: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(num_features,))
    x = layers.Dense(64, activation="relu")(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return AdversarialModel(inputs, outputs)


# encrypts model weights using Paillier encryption.
# This ensures secure communication with the server.


def encrypt_weights(weights: List[np.ndarray], pubkey) -> List[np.ndarray]:
    enc_layers = []
    for w in weights:
        flat_w = w.flatten()
        enc_vals = [pubkey.encrypt(float(val)) for val in flat_w]
        enc_arr = np.array(enc_vals, dtype=object).reshape(w.shape)
        enc_layers.append(enc_arr)
    return enc_layers


# Loading Data


def load_local_data(csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :NUM_FEATURES].values.astype(np.float32)
    y = df.iloc[:, NUM_FEATURES].values.astype(np.int32)
    return X, y


class IoTClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
    ):
        self.client_id = client_id
        # Build model & DP optimizer
        self.model = create_adversarial_model(NUM_FEATURES, NUM_CLASSES)
        self.dp_opt = DPKerasSGDOptimizer(
            l2_norm_clip=1.0,
            noise_multiplier=1.0,
            num_microbatches=32,
            learning_rate=0.01,
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=self.dp_opt,
            metrics=["accuracy"],
        )
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return local model weights
        return self.model.get_weights()

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # 1) Load global weights
        self.model.set_weights([w.astype(np.float32) for w in parameters])

        # 2) Parse round & DP noise
        round_num = int(config.get("round_number", "0"))
        noise_val = float(config.get("noise_multiplier", "1.0"))

        # 3) Snapshot old weights
        old_weights = self.model.get_weights()

        # --- PASS 1: RAW (no adv, no DP) ---
        from tensorflow.keras.optimizers import SGD
        # Build a clean model (same arch, no adversarial wrapper)
        clean_model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(NUM_FEATURES,)),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        clean_model.set_weights(old_weights)
        clean_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=SGD(learning_rate=0.01),
            metrics=["accuracy"],
        )
        clean_model.fit(self.x_train, self.y_train,
                        epochs=1, batch_size=32, verbose=0)
        new_raw_w   = clean_model.get_weights()
        raw_deltas  = [nr - ow for nr, ow in zip(new_raw_w, old_weights)]
        raw_flat    = flatten_weights(raw_deltas)
        raw_path    = f"captured_grads/raw_grad_client{self.client_id}_round{round_num}.npy"
        np.save(raw_path, raw_flat)
        print(f"[Client {self.client_id}] Saved raw grads → {raw_path}")

        # --- PASS 2: ADVERSARIAL (adv, no DP) ---
        # Re‐load old weights into our adversarial model
        self.model.set_weights(old_weights)
        adv_opt = DPKerasSGDOptimizer(
            l2_norm_clip=1.0,
            noise_multiplier=0.0,       # zero DP noise
            num_microbatches=32,
            learning_rate=0.01,
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=adv_opt,
            metrics=["accuracy"],
        )
        self.model.fit(self.x_train, self.y_train,
                       epochs=1, batch_size=32, verbose=0)
        new_adv_w   = self.model.get_weights()
        adv_deltas  = [na - ow for na, ow in zip(new_adv_w, old_weights)]
        adv_flat    = flatten_weights(adv_deltas)
        adv_path    = f"captured_grads/adv_grad_client{self.client_id}_round{round_num}.npy"
        np.save(adv_path, adv_flat)
        print(f"[Client {self.client_id}] Saved adversarial grads → {adv_path}")

        # --- PASS 3: ADV + DP NOISE (what we actually send) ---
        # Reset back to old weights again
        self.model.set_weights(old_weights)
        dp_opt = DPKerasSGDOptimizer(
            l2_norm_clip=1.0,
            noise_multiplier=noise_val,
            num_microbatches=32,
            learning_rate=0.01,
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=dp_opt,
            metrics=["accuracy"],
        )
        self.model.fit(self.x_train, self.y_train,
                       epochs=1, batch_size=32, verbose=0)
        new_dp_w    = self.model.get_weights()
        dp_deltas   = [nd - ow for nd, ow in zip(new_dp_w, old_weights)]
        dp_flat     = flatten_weights(dp_deltas)
        dp_path     = f"captured_grads/noisy_grad_client{self.client_id}_round{round_num}.npy"
        np.save(dp_path, dp_flat)
        print(f"[Client {self.client_id}] Saved DP‐noisy grads → {dp_path}")

        # 4) Local evaluation on the DP‐trained model
        train_loss, train_acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        test_loss,  test_acc  = self.model.evaluate(self.x_test,  self.y_test,  verbose=0)

        # 5) Save model snapshot
        model_path = f"client_models/client_{self.client_id}_round_{round_num}.h5"
        self.model.save(model_path, include_optimizer=False)
        print(f"[Client {self.client_id}] Saved local model → {model_path}")

        # 6) Return ONLY the DP‐noisy weights to the aggregator
        return (
            new_dp_w,
            len(self.x_train),
            {
                "client_id":      self.client_id,
                "train_accuracy": float(train_acc),
                "test_accuracy":  float(test_acc),
            },
        )

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float,int,Dict]:
        self.model.set_weights([w.astype(np.float32) for w in parameters])
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(acc), "client_id": self.client_id}


def main():
    parser = argparse.ArgumentParser(description="IoT Device for NSL-KDD (Flower 1.4.0)")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID (0,1,2...)")
    parser.add_argument("--data_file", type=str, required=True, help="CSV file for device data (e.g. device0.csv)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data for local test")
    args = parser.parse_args()

    # Load and split data
    X, y = load_local_data(args.data_file)
    split_idx = int(len(X) * (1 - args.test_split))
    train_data, test_data = (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])

    # Start Flower client
    client = IoTClient(str(args.client_id), train_data, test_data)
    print(f"[Device {args.client_id}] Starting Flower client…")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
