import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Adjust if needed for your data
NUM_FEATURES = 41  # 41 features + 1 label = 42 columns total.

# Example 5-class mapping
ATTACK_MAP_5CLASS = {
    "normal": 0,
    "neptune": 1,
    "smurf": 1,
    "pod": 1,
    "teardrop": 1,
    "land": 1,
    "back": 1,
    "apache2": 1,
    "udpstorm": 1,
    "mailbomb": 1,
    "processtable": 1,
    "satan": 2,
    "ipsweep": 2,
    "nmap": 2,
    "portsweep": 2,
    "mscan": 2,
    "saint": 2,
    "guess_passwd": 3,
    "ftp_write": 3,
    "imap": 3,
    "phf": 3,
    "multihop": 3,
    "warezmaster": 3,
    "warezclient": 3,
    "spy": 3,
    "xlock": 3,
    "xsnoop": 3,
    "snmpguess": 3,
    "snmpgetattack": 3,
    "httptunnel": 3,
    "sendmail": 3,
    "named": 3,
    "buffer_overflow": 4,
    "loadmodule": 4,
    "rootkit": 4,
    "perl": 4,
    "sqlattack": 4,
    "xterm": 4,
    "ps": 4,
}


def preprocess_kdd_data(train_file, test_file):
    #########################
    # 1. LOAD RAW DATA
    #########################
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    # 2. If shape is 43 columns, drop the last column (difficulty)
    if df_train.shape[1] == 43:
        df_train.drop(df_train.columns[-1], axis=1, inplace=True)
    if df_test.shape[1] == 43:
        df_test.drop(df_test.columns[-1], axis=1, inplace=True)

    #########################
    # 2. LABEL ENCODE CATEGORICAL COLUMNS (1, 2, 3)
    #########################
    label_encoders = {}
    categorical_columns = [1, 2, 3]

    for col in categorical_columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        label_encoders[col] = le  # Save encoder for potential inverse mapping

    #########################
    # 3. MAP ATTACK LABEL
    #########################
    df_train.iloc[:, -1] = df_train.iloc[:, -1].apply(
        lambda x: ATTACK_MAP_5CLASS.get(x.strip(), 0)
    )
    df_test.iloc[:, -1] = df_test.iloc[:, -1].apply(
        lambda x: ATTACK_MAP_5CLASS.get(x.strip(), 0)
    )

    #########################
    # 4. SCALE NUMERIC FEATURES
    #########################
    scaler = MinMaxScaler()
    X_train = df_train.iloc[:, :-1].values
    X_test = df_test.iloc[:, :-1].values

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Rebuild DataFrames with scaled features
    df_train_scaled = pd.DataFrame(X_train_scaled)
    df_train_scaled["attack_label"] = df_train.iloc[:, -1].values

    df_test_scaled = pd.DataFrame(X_test_scaled)
    df_test_scaled["attack_label"] = df_test.iloc[:, -1].values

    #########################
    # 5. SPLIT TRAINING DATA
    #########################
    num_devices = 3
    df_shuffled = df_train_scaled.sample(frac=1, random_state=42).reset_index(drop=True)
    splits = np.array_split(df_shuffled, num_devices)
    for i, splitdf in enumerate(splits):
        out_file = f"device{i}.csv"
        splitdf.to_csv(out_file, index=False)
        print(f"Saved {out_file} with {len(splitdf)} rows")

    #########################
    # 6. SAVE AGGREGATOR TEST
    #########################
    df_test_scaled.to_csv("aggregatorTest.csv", index=False)
    print("Saved aggregatorTest.csv for aggregator evaluation.")


if __name__ == "__main__":
    preprocess_kdd_data("KDDTrain+.txt", "KDDTest+.txt")
