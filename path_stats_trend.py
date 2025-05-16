import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import os
import numpy as np

# Regex to capture timestamp and key-value pairs from the trace log
# Adjust regex if log format slightly differs (e.g., timestamp format)
LOG_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\]\s+" # Timestamp
    r"(?:TRACE|DEBUG)\s+.*?]\s+" # Log level and source (more general)
    r"path stats:\s+"
    r"path_id=(?P<path_id>\d+)\s+"
    r"local_addr=(?P<local_addr>[\d.:\[\]]+)\s+" # Allow IPv6 addresses
    r"peer_addr=(?P<peer_addr>[\d.:\[\]]+)\s+"   # Allow IPv6 addresses
    r"validation_state=(?P<validation_state>\w+)\s+"
    r"state=(?P<state>\w+)\s+"
    r"recv=(?P<recv>\d+)\s+"
    r"sent=(?P<sent>\d+)\s+"
    r"lost=(?P<lost>\d+)\s+"
    r"lost_spurious=(?P<lost_spurious>\d+)\s+"
    r"retrans=(?P<retrans>\d+)\s+"
    r"rtt=(?P<rtt>[\d.]+)ms\s+"
    r"min_rtt=(?P<min_rtt>Some\(([\d.]+)ms\)|None)\s*"
    # Capture rttvar value and unit separately
    r"rttvar=(?P<rttvar_val>[\d.]+)(?P<rttvar_unit>µs|ms)\s*"
    r"rtt_update=(?P<rtt_update>\d+)\s+"
    r"cwnd=(?P<cwnd>\d+)\s+"
    r"sent_bytes=(?P<sent_bytes>\d+)\s+"
    r"recv_bytes=(?P<recv_bytes>\d+)\s+"
    r"lost_bytes=(?P<lost_bytes>\d+)\s+"
    r"stream_retrans_bytes=(?P<stream_retrans_bytes>\d+)\s+"
    r"pmtu=(?P<pmtu>\d+)\s+"
    r"delivery_rate=(?P<delivery_rate>\d+)"
    # Add more fields here if needed, ensuring spaces are handled
)

def parse_min_rtt(value_str):
    """Extracts float from 'Some(Xms)' or returns NaN for 'None'."""
    if value_str == 'None':
        return np.nan
    match = re.match(r"Some\(([\d.]+)ms\)", value_str)
    if match:
        return float(match.group(1))
    return np.nan

def parse_log_file(filepath):
    """Parses the log file and returns a list of dictionaries."""
    data = []
    first_timestamp = None
    with open(filepath, 'r') as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                record = match.groupdict()
                try:
                    # Convert types and handle units
                    record['path_id'] = int(record['path_id'])
                    record['recv'] = int(record['recv'])
                    record['sent'] = int(record['sent'])
                    record['lost'] = int(record['lost'])
                    record['lost_spurious'] = int(record['lost_spurious'])
                    record['retrans'] = int(record['retrans'])
                    record['rtt_ms'] = float(record['rtt']) # Already in ms
                    record['min_rtt_ms'] = parse_min_rtt(record['min_rtt']) # Special parser

                    # Handle rttvar based on unit
                    rttvar_val = float(record['rttvar_val'])
                    if record['rttvar_unit'] == 'µs':
                        record['rttvar_ms'] = rttvar_val / 1000.0 # Convert µs to ms
                    elif record['rttvar_unit'] == 'ms':
                         record['rttvar_ms'] = rttvar_val # Already in ms
                    else:
                         record['rttvar_ms'] = np.nan # Should not happen with regex

                    record['rtt_update'] = int(record['rtt_update'])
                    record['cwnd'] = int(record['cwnd'])
                    record['sent_bytes'] = int(record['sent_bytes'])
                    record['recv_bytes'] = int(record['recv_bytes'])
                    record['lost_bytes'] = int(record['lost_bytes'])
                    record['stream_retrans_bytes'] = int(record['stream_retrans_bytes'])
                    record['pmtu'] = int(record['pmtu'])
                    record['delivery_rate'] = int(record['delivery_rate']) # Bytes/sec

                    # Remove raw/intermediate fields
                    del record['rtt']
                    del record['min_rtt']
                    del record['rttvar_val']
                    del record['rttvar_unit']

                    data.append(record)
                except Exception as e:
                    print(f"Error processing record: {record} - {e}")
                    # print(f"Line: {line.strip()}") # Uncomment for debugging specific lines
    return data

def calculate_deltas_and_rates(df):
    """Calculates time deltas, byte deltas, and rates per path."""
    df = df.sort_values(by=['path_id', 'time_sec'])
    df['time_delta_sec'] = df.groupby('path_id')['time_sec'].diff()

    # Calculate deltas for cumulative counters
    cumulative_cols = ['sent_bytes', 'recv_bytes', 'sent', 'recv', 'lost', 'retrans', 'lost_bytes']
    for col in cumulative_cols:
        df[f'delta_{col}'] = df.groupby('path_id')[col].diff()

    # Calculate rates (handle division by zero or NaN time_delta)
    df['sending_rate_Bps'] = (df['delta_sent_bytes'] / df['time_delta_sec']).fillna(0)
    df['receiving_rate_Bps'] = (df['delta_recv_bytes'] / df['time_delta_sec']).fillna(0)

    # Calculate loss/retrans ratio (relative to packets sent in interval)
    # Avoid division by zero if no packets sent
    df['loss_ratio'] = (df['delta_lost'] / df['delta_sent'].replace(0, np.nan)).fillna(0)
    df['retrans_ratio'] = (df['delta_retrans'] / df['delta_sent'].replace(0, np.nan)).fillna(0)


    # Set rates to 0 where time delta is 0 or NaN to avoid inf/-inf
    df.loc[df['time_delta_sec'] <= 0, ['sending_rate_Bps', 'receiving_rate_Bps']] = 0

    return df

def plot_metrics(df, output_dir):
    """Generates and saves plots for key metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paths = df['path_id'].unique()
    paths.sort()

    # --- Plotting Function ---
    def save_plot(metric_y, title, ylabel, filename, log_scale=False, is_rate=False, smooth_window=None):
        plt.figure(figsize=(12, 6))
        for path_id in paths:
            path_df = df[df['path_id'] == path_id].sort_values('time_sec')
            y_data = path_df[metric_y]

            if smooth_window and len(y_data) > smooth_window:
                 # Apply rolling average only if enough data points exist
                 y_data = y_data.rolling(window=smooth_window, center=True, min_periods=1).mean()

            plt.plot(path_df['time_sec'], y_data, label=f'Path {path_id}', alpha=0.8)

        plt.xlabel("Time (seconds)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if log_scale:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # --- Generate Plots ---
    smoothing = 5 # Adjust smoothing window size (e.g., 5 points)

    # Performance Metrics
    save_plot('cwnd', 'Congestion Window (CWND) Over Time', 'CWND (Bytes)', 'cwnd_vs_time.png')
    save_plot('rtt_ms', 'Smoothed RTT Over Time', 'RTT (ms)', 'rtt_vs_time.png')
    save_plot('delivery_rate', 'Delivery Rate Over Time', 'Rate (Bytes/sec)', 'delivery_rate_vs_time.png')

    # Calculated Rates
    save_plot('sending_rate_Bps', f'Sending Rate Over Time (Smoothed {smoothing}pts)', 'Rate (Bytes/sec)', 'sending_rate_vs_time.png', smooth_window=smoothing)
    save_plot('receiving_rate_Bps', f'Receiving Rate Over Time (Smoothed {smoothing}pts)', 'Rate (Bytes/sec)', 'receiving_rate_vs_time.png', smooth_window=smoothing)

    # Cumulative Data
    save_plot('sent_bytes', 'Cumulative Sent Bytes Over Time', 'Bytes Sent', 'cumulative_sent_bytes.png')
    save_plot('recv_bytes', 'Cumulative Received Bytes Over Time', 'Bytes Received', 'cumulative_recv_bytes.png')

    # Loss/Retrans
    save_plot('loss_ratio', f'Packet Loss Ratio Over Time (Smoothed {smoothing}pts)', 'Loss Ratio (delta_lost / delta_sent)', 'loss_ratio_vs_time.png', smooth_window=smoothing)
    save_plot('retrans_ratio', f'Packet Retrans Ratio Over Time (Smoothed {smoothing}pts)', 'Retrans Ratio (delta_retrans / delta_sent)', 'retrans_ratio_vs_time.png', smooth_window=smoothing)
    save_plot('lost', 'Cumulative Lost Packets Over Time', 'Packets Lost', 'cumulative_lost_packets.png')


    print(f"Plots saved to directory: {output_dir}")


# --- Main Execution ---
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Parse and plot quiche path stats trace logs.')
    # parser.add_argument('logfile', help='Path to the server trace log file.')
    # parser.add_argument('-o', '--output', default='quiche_plots', help='Directory to save plot images.')
    # args = parser.parse_args()
    logfile = "/tmp/minitopo_experiences/quiche_client.log"
    output = "client"

    # print(f"Parsing log file: {args.logfile}")
    print(f"Parsing log file: {logfile}")
    parsed_data = parse_log_file(logfile)

    if not parsed_data:
        print("No valid 'path stats' log entries found.")
    else:
        df = pd.DataFrame(parsed_data)
        print(f"Parsed {len(df)} log entries.")

        print("Calculating deltas and rates...")
        df = calculate_deltas_and_rates(df)

        # Optional: Save processed data to CSV for further analysis
        # csv_path = os.path.join(args.output, 'processed_path_stats.csv')
        # df.to_csv(csv_path, index=False)
        # print(f"Processed data saved to {csv_path}")

        print(f"Generating plots in directory: {output}")
        plot_metrics(df, output)

        print("Analysis complete.")