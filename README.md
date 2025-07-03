# IoT FL IDS Project

A comprehensive guide to run both the **insecure** and **secure (federated learning)** intrusion detection systems in a Windows environment using VS Code.  

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Insecure System](#running-the-insecure-system)
   - [Data Preprocessing](#data-preprocessing)
   - [Launching Components (No Malicious Files)](#launching-components-no-malicious-files)
   - [Launching Components (With Malicious Files)](#launching-components-with-malicious-files)
4. [Real-World Data Injection](#real-world-data-injection)
5. [Running the Secure System (Federated Learning)](#running-the-secure-system-federated-learning)
6. [Gradient Inversion Attack Simulation](#gradient-inversion-attack-simulation)
7. [Directory Structure](#directory-structure)
8. [Notes](#notes)

---

## Prerequisites

- **Operating System:** Windows 10 or later (Admin privileges required for secure system).  
- **IDE:** Visual Studio Code (with integrated terminals).  
- **Python:** 3.8+ installed and added to `PATH`.  
- **Npcap:** Installed with "Loopback Adapter" support. Required for passive MITM sniffing.  
- **VS Code Extensions:** Python extension (optional but recommended).  

---

## Installation

1. **Install Npcap**  
   Download and install from [nmap.org/npcap](https://nmap.org/npcap/). Ensure "Install Npcap in WinPcap API-compatible Mode" is **checked**.  
2. **Clone the repository** and open it in VS Code.  
3. **Install Python dependencies** in your project environment:  
   ```powershell
   pip install -r requirements.txt
   ```

---

## Running the Insecure System

> Implements a passive MITM (Raw Infiltration) attack alongside a basic federated learning server–client flow without differential privacy.  

### Data Preprocessing

1. Open a new terminal in VS Code (`Terminal → New Terminal`).  
2. Navigate to the `insecure_system` folder:  
   ```powershell
   cd insecure_system
   ```
3. Generate device splits and test set:  
   ```powershell
   python data_preprocessing.py
   ```

### Launching Components (No Malicious Files)

Open **five** separate terminals in VS Code, labeled **A**, **B**, **C**, **D**, and **E**.

**Terminal E: Start the MITM proxy**  
```powershell
mitmdump --mode reverse:http://127.0.0.1:5000 --listen-port 8888 -w mitm.flows
```

**Terminals B, C, D: Configure proxy environment**  
```powershell
Remove-Item Env:\HTTP_PROXY  -ErrorAction SilentlyContinue
Remove-Item Env:\HTTPS_PROXY -ErrorAction SilentlyContinue

$Env:HTTP_PROXY  = "http://127.0.0.1:8888"
$Env:HTTPS_PROXY = "http://127.0.0.1:8888"
```

**Terminal A: Start the aggregator**  
```powershell
python insecure_aggregator.py --host 127.0.0.1 --port 5000
```

**Terminals B, C, D: Start the devices**  
```powershell
# Terminal B
python insecure_device.py --client_id=0 --data_file=device0.csv

# Terminal C
python insecure_device.py --client_id=1 --data_file=device1.csv

# Terminal D
python insecure_device.py --client_id=2 --data_file=device2.csv
```

**Terminal E: Capture and save proxy logs**  
```powershell
mitmdump -nr mitm.flows -v > mitm.log
```

### Launching Components (With Malicious Files)

Repeat the **Launching Components** steps above, but in **Terminal B** replace the client command with:  
```powershell
python insecure_device.py --client_id=0 --data_file=malicious.csv
```

---

## Real-World Data Injection

Inject synthetic samples into a device CSV (e.g., `device2.csv`):  
```powershell
python inject_dynamic_data.py   --device_file device2.csv   --allow_low_conf
```

---

## Running the Secure System (Federated Learning)

> Implements full FL with Differential Privacy and adversarial training.  

1. Open a new terminal and **run as Administrator**.  
2. Navigate to the `secure_system` folder and create the `captured_grads` directory:  
   ```powershell
   cd secure_system
   mkdir captured_grads
   ```
3. Preprocess data (same as insecure):  
   ```powershell
   python data_preprocessing.py
   ```
4. **Terminal A**: Start the FL aggregator with noise choice:  
   ```powershell
   # Gaussian noise
   python aggregator2.py --noise gaussian

   # Laplace noise
   python aggregator2.py --noise laplace
   ```
5. **Terminal B**: Start the passive sniffer (attacker):  
   ```powershell
   python attacker.py
   ```
6. **Terminals C, D, E**: Start the FL devices:  
   ```powershell
   python device.py --client_id=0 --data_file=device0.csv
   python device.py --client_id=1 --data_file=device1.csv
   python device.py --client_id=2 --data_file=device2.csv
   ```
7. **With Malicious File** (optional):  
   ```powershell
   python device.py --client_id=0 --data_file=malicious.csv
   ```

---

## Gradient Inversion Attack Simulation

Reconstruct features from captured gradients:  

```powershell
# Raw gradient inversion
python inversion_attack.py   --grad_file captured_grads/raw_grad_client0_round1.npy   --data_file device0.csv

# DP-noisy gradient inversion
python inversion_attack.py   --grad_file captured_grads/noisy_grad_client0_round1.npy   --dp   --data_file device0.csv

# Summarize and plot results
python summarize_inversion_and_plot.py
```

---

## Directory Structure

```
project-root/
├─ insecure_system/
│  ├─ data_preprocessing.py
│  ├─ insecure_aggregator.py
│  ├─ insecure_device.py
│  ├─ inject_dynamic_data.py
│  ├─ malicious.csv
│  ├─ device0.csv
│  ├─ device1.csv
│  ├─ device2.csv
│  └─ aggregatorTest.csv
├─ secure_system/
│  ├─ data_preprocessing.py
│  ├─ aggregator2.py
│  ├─ device.py
│  ├─ attacker.py
│  ├─ inversion_attack.py
│  ├─ summarize_inversion_and_plot.py
│  └─ captured_grads/
├─ requirements.txt
└─ README.md
```

---

## Notes

- All `python` commands assume your environment points to the correct interpreter.  
- Ensure **Npcap** loopback adapter is installed before running `attacker.py`.  
- Administrator privileges are required for packet sniffing and secure FL components.  
- Logs and model snapshots are saved in their respective folders (`plots/`, `aggregator_models/`, `client_models/`, etc.).
