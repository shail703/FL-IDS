import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Adjust if needed for your data
NUM_FEATURES = 41  # 41 features + 1 label = 42 columns total.

# 5-class mapping for NSL-KDD labels
ATTACK_MAP_5CLASS = {
    "normal": 0,
    # DOS attacks → class 1
    "neptune": 1, "smurf": 1, "pod": 1, "teardrop": 1, "land": 1,
    "back":    1, "apache2": 1, "udpstorm": 1, "mailbomb": 1, "processtable": 1,
    # PROBE attacks → class 2
    "satan": 2, "ipsweep": 2, "nmap": 2, "portsweep": 2, "mscan": 2, "saint": 2,
    # R2L attacks → class 3
    "guess_passwd": 3, "ftp_write": 3, "imap": 3, "phf": 3, "multihop": 3,
    "warezmaster":3, "warezclient":3, "spy":3, "xlock":3, "xsnoop":3,
    "snmpguess":3, "snmpgetattack":3, "httptunnel":3, "sendmail":3,
    "named":3,
    # U2R attacks → class 4
    "buffer_overflow":4, "loadmodule":4, "rootkit":4,
    "perl":4, "sqlattack":4, "xterm":4, "ps":4,
}

def preprocess_kdd_data(
    train_file: str,
    test_file: str,
    num_devices: int = 3
):
    # 1. Load raw data
    df_train = pd.read_csv(train_file, header=None)
    df_test  = pd.read_csv(test_file,  header=None)

    # Drop difficulty column if present
    if df_train.shape[1] == 43:
        df_train.drop(df_train.columns[-1], axis=1, inplace=True)
    if df_test.shape[1] == 43:
        df_test.drop(df_test.columns[-1], axis=1, inplace=True)

    # 2. Encode categorical cols (protocol_type, service, flag)
    for col in [1, 2, 3]:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        df_test[col]  = le.transform(df_test[col].astype(str))

    # 3. Map labels to 5 classes
    df_train.iloc[:, -1] = (
        df_train.iloc[:, -1]
        .str.strip()
        .map(ATTACK_MAP_5CLASS)
        .fillna(0)
        .astype(int)
    )
    df_test.iloc[:, -1] = (
        df_test.iloc[:, -1]
        .str.strip()
        .map(ATTACK_MAP_5CLASS)
        .fillna(0)
        .astype(int)
    )

    # 4. Scale numeric features to [0,1]
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(df_train.iloc[:, :-1])
    X_te = scaler.transform(df_test.iloc[:, :-1])

    df_train_scaled = pd.DataFrame(X_tr)
    df_train_scaled['attack_label'] = df_train.iloc[:, -1].values

    df_test_scaled = pd.DataFrame(X_te)
    df_test_scaled['attack_label'] = df_test.iloc[:, -1].values

    # 5. Split training among devices
    df_shuf = df_train_scaled.sample(frac=1, random_state=42).reset_index(drop=True)
    splits = np.array_split(df_shuf, num_devices)
    for i, split in enumerate(splits):
        out_file = f"device_{i}.csv"
        split.to_csv(out_file, index=False)
        print(f"Saved {out_file} with {len(split)} rows")

    # 6. Save aggregator test set
    df_test_scaled.to_csv("aggregatorTest.csv", index=False)
    print("Saved aggregatorTest.csv for central evaluation.")

if __name__ == "__main__":
    preprocess_kdd_data("KDDTrain+.txt", "KDDTest+.txt")
