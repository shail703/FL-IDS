#!/usr/bin/env python
"""
attacker_sniffer.py
───────────────────
Passive MITM logger (Windows).  Captures every TCP payload on localhost
port 8080 (Flower).  Requires:

  • Npcap with Loopback Adapter   • pip install scapy   • Admin shell
"""
import datetime
import os
import sys
import ctypes

from scapy.all import sniff, TCP, Raw, get_if_list, conf
from scapy.layers.inet import IP

FLOW_PORT = 8080
LOG_PATH  = "attacker_log.txt"

# ── pick Npcap loopback adapter ───────────────────────────────────────────────
def pick_loop_iface():
    for name in get_if_list():
        if "loopback" in name.lower():
            return name
    return conf.iface

IFACE = pick_loop_iface()

# ── helpers ───────────────────────────────────────────────────────────────────
def hexdump(buf: bytes, cols: int = 16) -> str:
    out = []
    for i in range(0, len(buf), cols):
        chunk = buf[i : i + cols]
        hx  = " ".join(f"{b:02x}" for b in chunk)
        txt = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"{i:04x}  {hx:<{cols*3}}  {txt}")
    return "\n".join(out)

def log(direction: str, payload: bytes):
    ts = datetime.datetime.now().isoformat(timespec="milliseconds")
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    # explicit UTF-8 so Unicode arrows/ellipses always encode
    with open(LOG_PATH, "a", encoding="utf-8", buffering=1) as fp:
        fp.write(f"\n[{ts}]  {direction}  {len(payload)} bytes\n")
        fp.write(hexdump(payload) + "\n")

def handle(pkt):
    if not pkt.haslayer(Raw):
        return
    tcp: TCP = pkt[TCP]
    data: bytes = bytes(pkt[Raw].load)

    direction = (f"{pkt[IP].src}:{tcp.sport} -> Aggregator"
                 if tcp.dport == FLOW_PORT
                 else f"Aggregator -> {pkt[IP].dst}:{tcp.dport}")

    log(direction, data)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[ATTACKER] Interface : {IFACE}")
    print(f"[ATTACKER] Capturing TCP port {FLOW_PORT} ...  (CTRL-C to stop)")
    print(f"[ATTACKER] Log file  : {LOG_PATH}\n")

    sniff(
        iface=IFACE,
        store=False,
        prn=handle,
        lfilter=lambda p: p.haslayer(TCP)
                         and (p[TCP].sport == FLOW_PORT or p[TCP].dport == FLOW_PORT),
    )

if __name__ == "__main__":
    if os.name == "nt" and not ctypes.windll.shell32.IsUserAnAdmin():
        sys.exit("Run this script from an **Administrator** terminal.")
    main()
