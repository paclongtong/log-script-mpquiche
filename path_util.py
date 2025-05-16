import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_metrics_qlog(path):
    """
    Parse recovery:metrics_updated events for time (in ms), path_id, bytes_in_flight, congestion_window.
    """
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('name') != 'recovery:metrics_updated':
                continue
            data = ev.get('data', {})
            t = ev.get('time')
            if t is None:
                continue
            rec = {
                'time': t,
                'path_id': data.get('path_id'),
                'bytes_in_flight': data.get('bytes_in_flight')
            }
            # add congestion_window if present
            if 'congestion_window' in data:
                rec['cwnd'] = data.get('congestion_window')
            records.append(rec)
    return pd.DataFrame(records)


def compute_utilization(df):
    """
    Given a metrics DataFrame with columns time (ms), path_id, bytes_in_flight, cwnd,
    forward-fill missing cwnd and bytes_in_flight per path,
    and compute utilization = bytes_in_flight / cwnd.
    Returns DataFrame with time_ms, path_id, util.
    """
    if df.empty:
        return pd.DataFrame(columns=['time_ms', 'path_id', 'util'])

    # sort by absolute time
    df = df.sort_values('time').reset_index(drop=True)

    # align to zero at first event
    zero = df['time'].iloc[0]
    df['time_ms'] = df['time'] - zero  # time is already in ms

    # forward-fill missing fields per path
    df['cwnd'] = df.groupby('path_id')['cwnd'].ffill()
    df['bytes_in_flight'] = df.groupby('path_id')['bytes_in_flight'].ffill()

    # compute utilization ratio
    df['util'] = df['bytes_in_flight'] / df['cwnd']

    return df[['time_ms', 'path_id', 'util']]


def plot_runs(runs):
    """
    Plot per-path utilization for each run.
    runs: list of tuples (label, df_util)
    """
    plt.figure(figsize=(10, 6))
    for label, df in runs:
        for pid, grp in df.groupby('path_id'):
            plt.plot(grp['time_ms'], grp['util'], label=f"{label}-path{pid}")

    plt.title('Per-Path Utilization Ratio Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Utilization (bytes_in_flight / cwnd)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot per-path utilization ratio from one or more QLOGs')
    parser.add_argument(
        '--run', '-r', nargs=2, action='append', metavar=('LABEL', 'QLOG_PATH'),
        required=True,
        help='Specify a run: LABEL QLOG_PATH. Can be repeated for multiple runs.'
    )
    args = parser.parse_args()

    runs = []
    for label, path in args.run:
        df_metrics = parse_metrics_qlog(path)
        df_util = compute_utilization(df_metrics)
        runs.append((label, df_util))

    plot_runs(runs)
