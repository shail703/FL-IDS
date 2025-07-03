#!/usr/bin/env python
"""
aggregator2.py
--------------

Server (aggregator) side code for a federated 5-class IDS using NSL-KDD.

* Uses coordinate-wise median to aggregate client updates.
* Optionally adds DP noise (Gaussian or Laplace) to the aggregated update.
* Evaluates global model on aggregatorTest.csv each round (41 features + 1 label => 5 classes).
* Saves aggregator model to 'aggregator_models/agg_model_round_{r}.h5' every round.
* Automatically generates graphs (client training accuracies, client testing accuracies,
  client validation losses, aggregator accuracy, aggregator loss) after training finishes.

Usage:
    python aggregator2.py --noise {gaussian,laplace}
"""

import argparse
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
from phe import paillier, EncryptedNumber
import os
import time
from tensorflow import keras
from tensorflow.keras import layers
from flwr.common import (
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import re
import matplotlib.pyplot as plt

# Constants and data loading
NUM_FEATURES = 41
NUM_CLASSES = 5
AGG_TEST_FILE = "aggregatorTest.csv"

# Load aggregator test set
df_agg_test = pd.read_csv(AGG_TEST_FILE)
X_val_agg = df_agg_test.iloc[:, :NUM_FEATURES].values.astype(np.float32)
y_val_agg = df_agg_test.iloc[:, NUM_FEATURES].values.astype(np.int32)

def build_keras_model() -> keras.Model:
    tf.keras.backend.clear_session()
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(NUM_FEATURES,)),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"]
    )
    return model

def coordinate_wise_median(all_updates: List[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(all_updates)
    return np.median(stacked, axis=0)

def unflatten_weights(flat: np.ndarray, shapes: List[Tuple[int]]) -> List[np.ndarray]:
    """Convert a flat 1D array back into a list of weight arrays of given shapes."""
    weights = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(flat[idx: idx + size].reshape(shape))
        idx += size
    return weights

# Generate a Paillier key pair (for demonstration)
print("[Aggregator] Generating Paillier key pair with random seed=999 for encryption demos.")
np.random.seed(999)
_pub, _priv = paillier.generate_paillier_keypair(n_length=1024)
public_key = paillier.PaillierPublicKey(_pub.n)
private_key = _priv

# ========================
# GRAPH PLOTTING FUNCTION
# ========================
def plot_aggregator_graphs(log_lines: List[str]):
    output_dir = "aggregator_graphs"
    os.makedirs(output_dir, exist_ok=True)

    client_accs = []
    client_tests = []
    client_vals = []
    agg_evals = []

    for line in log_lines:
        # Client train/test accuracy from fit
        m = re.match(
            r"\[Aggregator\]\[Round (\d+)\] Client (\d+): train_acc=([\d.]+), test_acc=([\d.]+)", 
            line
        )
        if m:
            rnd, cid, tr, te = m.groups()
            client_accs.append({
                "round": int(rnd),
                "client_id": int(cid),
                "train_acc": float(tr)
            })
            client_tests.append({
                "round": int(rnd),
                "client_id": int(cid),
                "test_acc": float(te)
            })

        # Client validation loss from evaluate (capture full device ID)
        m = re.match(
            r"\[Aggregator\]\[Round (\d+)\] device ([^ ]+) => val_loss=([\d.]+), #examples=(\d+)",
            line
        )
        if m:
            rnd, cid, vl, n = m.groups()
            client_vals.append({
                "round": int(rnd),
                "client_id": cid,
                "val_loss": float(vl)
            })

        # Aggregator test eval
        m = re.match(
            r"\[Aggregator\]\[Round (\d+)\] aggregatorTest => loss=([\d.]+), acc=([\d.]+)", 
            line
        )
        if m:
            rnd, loss, acc = m.groups()
            agg_evals.append({
                "round": int(rnd),
                "loss": float(loss),
                "acc": float(acc)
            })

    df_train = pd.DataFrame(client_accs)
    df_test  = pd.DataFrame(client_tests)
    df_val   = pd.DataFrame(client_vals)
    df_agg   = pd.DataFrame(agg_evals)

    # Plot client training accuracy
    plt.figure(figsize=(10, 6))
    for cid in df_train["client_id"].unique():
        subset = df_train[df_train["client_id"] == cid]
        plt.plot(subset["round"], subset["train_acc"], label=f"Client {cid}")
    plt.xlabel("Round")
    plt.ylabel("Training Accuracy")
    plt.title("Client Training Accuracy per Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "client_training_accuracy.png"))
    plt.close()

    # Plot client testing accuracy
    plt.figure(figsize=(10, 6))
    for cid in df_test["client_id"].unique():
        subset = df_test[df_test["client_id"] == cid]
        plt.plot(subset["round"], subset["test_acc"], label=f"Client {cid}")
    plt.xlabel("Round")
    plt.ylabel("Testing Accuracy")
    plt.title("Client Testing Accuracy per Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "client_testing_accuracy.png"))
    plt.close()

    # Plot client validation loss with Noise σ labels
    rounds = sorted(df_val["round"].unique())
    noise_vals = [max(0.25 - 0.02*(r - 1), 0.1) for r in rounds]
    plt.figure(figsize=(10, 6))
    for cid in df_val["client_id"].unique():
        subset = df_val[df_val["client_id"] == cid]
        plt.plot(subset["round"], subset["val_loss"], marker='o', label=f"{cid}")
    plt.xlabel("Round")
    plt.ylabel("Validation Loss")
    plt.title("Client Validation Loss per Round (with Noise σ)")
    # Annotate x-ticks with round and its noise multiplier
    labels = [f"R{r}\\nσ={noise_vals[i]:.2f}" for i, r in enumerate(rounds)]
    plt.xticks(rounds, labels, rotation=45, ha="right")
    plt.legend(title="Client ID")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "client_validation_loss_with_noise.png"))
    plt.close()

    # Plot aggregator accuracy with Noise σ labels
    rounds = sorted(df_agg["round"].unique())
    noise_vals = [max(0.25 - 0.02*(r - 1), 0.1) for r in rounds]

    plt.figure(figsize=(10, 6))
    plt.plot(df_agg["round"], df_agg["acc"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Aggregator Accuracy per Round (with Noise σ)")

    # Annotate x-ticks with round and its noise multiplier
    labels = [f"R{r}\nσ={noise_vals[i]:.2f}" for i, r in enumerate(rounds)]
    plt.xticks(rounds, labels, rotation=45, ha="right")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregator_accuracy_with_noise.png"))
    plt.close()


    # Plot aggregator loss
    plt.figure(figsize=(10, 6))
    plt.plot(df_agg["round"], df_agg["loss"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Aggregator Loss over Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "aggregator_loss.png"))
    plt.close()

    print(f"Graph plots saved in folder: {output_dir}")

# ================================
# AdvancedAggregator with DP noise
# ================================
class AdvancedAggregator(Strategy):
    def __init__(
        self,
        noise_type: str = 'gaussian',
        total_rounds: int = 25,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        self.noise_type = noise_type
        self.total_rounds = total_rounds
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.current_global_weights: Optional[List[np.ndarray]] = None
        self.round_logs: List[str] = []
        os.makedirs("aggregator_models", exist_ok=True)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        init_model = build_keras_model()
        init_weights = init_model.get_weights()
        self.current_global_weights = [w.astype(np.float32) for w in init_weights]
        return ndarrays_to_parameters(self.current_global_weights)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        print(f"[Aggregator] Round {server_round} => waiting up to 15s for enough clients to be available.")
        self.round_logs.append(f"[Aggregator] Round {server_round} => waiting up to 15s for enough clients to be available.")
        time.sleep(15)
        all_clients = list(client_manager.clients.values())
        if len(all_clients) < self.min_available_clients:
            log = "[Aggregator] Not enough clients => skipping fit."
            print(log)
            self.round_logs.append(log)
            return []
        noise_val = max(0.25 - 0.02*(server_round - 1), 0.1)
        fit_config = {
            "noise_multiplier": str(noise_val),
            "round_number": str(server_round),
        }
        return [(c, FitIns(parameters, fit_config)) for c in all_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        if not results and self.current_global_weights is not None:
            return ndarrays_to_parameters(self.current_global_weights), {}

        client_flats: List[np.ndarray] = []
        shapes: Optional[List[Tuple[int]]] = None

        for (client_proxy, fit_res) in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            if shapes is None:
                shapes = [arr.shape for arr in nds]
            flat_vals = []
            for arr_enc in nds:
                arr = np.array([
                    private_key.decrypt(v) if isinstance(v, EncryptedNumber) else v
                    for v in arr_enc.flatten()
                ], dtype=np.float32)
                flat_vals.append(arr)
            client_flats.append(np.concatenate(flat_vals))

            cid = fit_res.metrics.get("client_id", client_proxy.cid)
            tr = fit_res.metrics.get("train_accuracy", "N/A")
            te = fit_res.metrics.get("test_accuracy", "N/A")
            log = f"[Aggregator][Round {server_round}] Client {cid}: train_acc={tr}, test_acc={te}"
            print(log)
            self.round_logs.append(log)

        aggregated_flat = coordinate_wise_median(client_flats)
        noise_val = max(0.25 - 0.02*(server_round - 1), 0.1)
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, noise_val, size=aggregated_flat.shape)
        elif self.noise_type == 'laplace':
            noise = np.random.laplace(0, noise_val, size=aggregated_flat.shape)
        else:
            noise = np.zeros_like(aggregated_flat)
        aggregated_flat += noise

        new_weights = unflatten_weights(aggregated_flat, shapes) if shapes else []
        self.current_global_weights = new_weights

        model = build_keras_model()
        model.set_weights(new_weights)
        save_path = f"aggregator_models/agg_model_round_{server_round}.h5"
        model.save(save_path)
        loss_val, acc_val = model.evaluate(X_val_agg, y_val_agg, verbose=0)
        log = f"[Aggregator][Round {server_round}] aggregatorTest => loss={loss_val:.4f}, acc={acc_val:.4f}"
        print(log)
        self.round_logs.append(log)

        metrics = {
            "aggregator_loss": float(loss_val),
            "aggregator_accuracy": float(acc_val),
        }
        return ndarrays_to_parameters(new_weights), metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        clients = list(client_manager.clients.values())
        if not clients:
            return []
        eval_config = {"round_number": str(server_round)}
        return [(c, EvaluateIns(parameters, eval_config)) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, Scalar]]:
        if not results:
            log = f"[Aggregator][Round {server_round}] No client evaluate results."
            print(log)
            self.round_logs.append(log)
            return 0.0, {}
        total_loss, total_samples = 0.0, 0
        for (cp, ev) in results:
            loss, n = ev.loss, ev.num_examples
            log = f"[Aggregator][Round {server_round}] device {cp.cid} => val_loss={loss}, #examples={n}"
            print(log)
            self.round_logs.append(log)
            total_loss += loss * n
            total_samples += n
        avg_loss = total_loss / total_samples if total_samples else 0.0
        return avg_loss, {"client_eval_loss": float(avg_loss)}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = build_keras_model()
        model.set_weights(parameters_to_ndarrays(parameters))
        loss_val, acc_val = model.evaluate(X_val_agg, y_val_agg, verbose=0)
        return float(loss_val), {"agg_eval_acc": float(acc_val)}

# ================================
# Entry point
# ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise",
        choices=["gaussian", "laplace"],
        default="gaussian",
        help="Type of DP noise to add in aggregation."
    )
    args = parser.parse_args()

    strategy = AdvancedAggregator(
        noise_type=args.noise,
        total_rounds=25,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    print(f"[Aggregator] Starting Flower server on 127.0.0.1:8080 using '{args.noise}' noise.")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=strategy.total_rounds),
    )
    plot_aggregator_graphs(strategy.round_logs)


if __name__ == "__main__":
    main()
