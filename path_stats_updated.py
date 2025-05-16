#!/usr/bin/env python3
import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

LOG_PATTERN = re.compile(
    r'\['
    r'(?P<prefix>[^]]+)'              # inside brackets: timestamp + level + module
    r'\].*?path stats:\s*'
    r'(?P<body>.*)$'                  # remainder of line after "path stats:"
)

INT_FIELDS = {
    'path_id', 'recv', 'sent', 'lost', 'lost_spurious',
    'retrans', 'rtt_update', 'cwnd', 'sent_bytes',
    'recv_bytes', 'lost_bytes', 'stream_retrans_bytes',
    'pmtu', 'delivery_rate'
}
MS_FIELDS = {'rtt', 'min_rtt', 'rttvar'}

def parse_log(filename):
    records = []
    with open(filename, 'r') as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            raw_prefix = m.group('prefix')
            ts_str = raw_prefix.split()[0].replace('Z', '+00:00')
            try:
                dt = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            body = m.group('body')
            kv_pairs = re.findall(r'(\w+)=([^\s,]+)', body)
            rec = {'timestamp': dt}
            for key, raw in kv_pairs:
                val = raw
                if key in MS_FIELDS:
                    if raw.endswith('ms'):
                        # already in milliseconds
                        rec[key] = float(raw[:-2])
                    elif raw.endswith('µs') or raw.endswith('us'):
                        # microseconds → milliseconds
                        micros = float(raw[:-2])
                        rec[key] = micros / 1000.0
                    else:
                        # fallback to float
                        rec[key] = float(raw)
                elif key in INT_FIELDS:
                    val_clean = val[:-2] if val.endswith('ms') else val
                    try:
                        rec[key] = int(val_clean)
                    except ValueError:
                        rec[key] = val_clean
                else:
                    rec[key] = val
            records.append(rec)
    if not records:
        print(f"No path-stats debug lines found in {filename}", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(records)
    df.sort_values('timestamp', inplace=True)
    return df

def parse_log_1(filename):
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue

            # --- timestamp parse (unchanged) ---
            raw_prefix = m.group('prefix')
            ts_str = raw_prefix.split()[0].replace('Z', '+00:00')
            try:
                dt = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            body = m.group('body')
            kv_pairs = re.findall(r'(\w+)=([^\s,]+)', body)

            rec = {'timestamp': dt}
            for key, raw in kv_pairs:
                # strip Some(...) wrapper
                if raw.startswith('Some(') and raw.endswith(')'):
                    raw = raw[5:-1]
                # literal None → None
                if raw == 'None':
                    rec[key] = None
                    continue

                # 1) millisecond/microsecond timing fields
                if key in MS_FIELDS:
                    if raw.endswith('ms'):
                        rec[key] = float(raw[:-2])
                    elif raw.endswith('µs') or raw.endswith('us'):
                        rec[key] = float(raw[:-2]) / 1000.0
                    else:
                        rec[key] = float(raw)

                # 2) integer counters
                elif key in INT_FIELDS:
                    v = raw[:-2] if raw.endswith('ms') else raw
                    try:
                        rec[key] = int(v)
                    except ValueError:
                        rec[key] = v

                # 3) everything else (addrs, states…)
                else:
                    rec[key] = raw

            records.append(rec)

    if not records:
        print(f"No path-stats debug lines found in {filename}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    df.sort_values('timestamp', inplace=True)
    return df


def plot_trends(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # RTT over time
    fig = plt.figure(figsize=(10, 4))
    for pid, group in df.groupby('path_id'):
        plt.plot(group['timestamp'], group['rtt'], label=f'Path {pid}')
    plt.xlabel('Time')
    plt.ylabel('RTT (ms)')
    plt.title('Per-Path RTT over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
    rtt_path = os.path.join(output_dir, 'rtt_trend.png')
    fig.savefig(rtt_path)
    print(f"Saved RTT trend to {rtt_path}")

    # Delivery rate over time
    fig = plt.figure(figsize=(10, 4))
    for pid, group in df.groupby('path_id'):
        plt.plot(group['timestamp'], group['delivery_rate'], label=f'Path {pid}')
    plt.xlabel('Time')
    plt.ylabel('Delivery Rate (bytes/s)')
    plt.title('Per-Path Delivery Rate over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
    dr_path = os.path.join(output_dir, 'delivery_rate_trend.png')
    fig.savefig(dr_path)
    print(f"Saved delivery rate trend to {dr_path}")

    # Congestion window over time
    if 'cwnd' in df.columns:
        fig = plt.figure(figsize=(10, 4))
        for pid, group in df.groupby('path_id'):
            plt.plot(group['timestamp'], group['cwnd'], label=f'Path {pid}')
        plt.xlabel('Time')
        plt.ylabel('Congestion Window (bytes)')
        plt.title('Per-Path Congestion Window over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
        cwnd_path = os.path.join(output_dir, 'cwnd_trend.png')
        fig.savefig(cwnd_path)
        print(f"Saved cwnd trend to {cwnd_path}")

    # 4) Lost and lost_spurious over time
    fig = plt.figure(figsize=(10, 4))
    for pid, group in df.groupby('path_id'):
        plt.plot(group['timestamp'], group['lost'],          label=f'Path {pid} lost')
        plt.plot(group['timestamp'], group['lost_spurious'], linestyle='--', label=f'Path {pid} spurious')
    plt.xlabel('Time')
    plt.ylabel('Packet Loss Count')
    plt.title('Per-Path Lost & Spurious Loss over Time')
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_trend.png'))
    plt.show()
    print(f"Saved loss trend to {output_dir}/loss_trend.png")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} LOG_FILE", file=sys.stderr)
        sys.exit(1)
    log_file = sys.argv[1]
    df = parse_log_1(log_file)
    print(df[['timestamp','path_id','rtt','delivery_rate','cwnd']].head(10).to_string(index=False))
    # determine client folder beside the log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    output_dir = os.path.join(log_dir, 'server')
    plot_trends(df, output_dir)

if __name__ == '__main__':
    main()

