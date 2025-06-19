import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re

# --- Configuration ---
BBR_LOG_FILE = "/tmp/minitopo_experiences/bbr_log.csv"
OUTPUT_DIR = "./bbr_plots/" # Make sure this directory exists or create it
TIME_COLUMN = "timestamp"
PATH_ID_COLUMN = "path_id"

# Ensure output directory exists
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Load Data ---
def parse_bbr_line(line):
    """Parse a single line of BBR log data with complex timestamp format"""
    line = line.strip()
    if not line:
        return None
    
    # Extract timestamp part (everything from "Instant" to the closing "}")
    timestamp_match = re.search(r'Instant \{ tv_sec: \d+, tv_nsec: \d+ \}', line)
    if not timestamp_match:
        return None
    
    timestamp_str = timestamp_match.group(0)
    # Get the rest of the line after the timestamp (skip the comma after })
    rest_of_line = line[timestamp_match.end():].lstrip(',').strip()
    
    # Split the rest by commas
    parts = [p.strip() for p in rest_of_line.split(',')]
    if len(parts) < 15:  # We expect at least 15 fields after timestamp
        return None
    
    return [timestamp_str] + parts

try:
    # Read the file line by line and parse manually
    data_rows = []
    with open(BBR_LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    # Skip header line (first line)
    for line in lines[1:]:
        parsed_row = parse_bbr_line(line)
        if parsed_row:
            data_rows.append(parsed_row)
    
    # Define column names
    column_names = [
        'timestamp', 'path_id', 'bbr_mode', 'btl_bw', 'pacing_rate', 
        'delivery_rate_estimate', 'min_rtt_us', 'latest_rtt_us', 'cwnd', 
        'bytes_in_flight', 'pacing_gain', 'cwnd_gain', 'cycle_idx', 
        'newly_lost_bytes', 'is_app_limited', 'app_limited_since'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    
except FileNotFoundError:
    print(f"Error: BBR log file not found at {BBR_LOG_FILE}")
    exit()
except Exception as e:
    print(f"Error reading BBR log file: {e}")
    exit()

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"First few timestamp values: {df[TIME_COLUMN].head().tolist()}")

# --- Data Preprocessing ---
def parse_instant_timestamp(timestamp_str):
    """Parse timestamp in format 'Instant { tv_sec: X, tv_nsec: Y }'"""
    if pd.isna(timestamp_str):
        return np.nan
    
    timestamp_str = str(timestamp_str)
    
    # Handle the Instant format with commas
    if 'Instant' in timestamp_str and 'tv_sec' in timestamp_str:
        # Extract tv_sec and tv_nsec using regex (note the comma between them)
        match = re.search(r'tv_sec:\s*(\d+),\s*tv_nsec:\s*(\d+)', timestamp_str)
        if match:
            tv_sec = int(match.group(1))
            tv_nsec = int(match.group(2))
            return tv_sec + tv_nsec / 1e9
        else:
            print(f"Failed to parse timestamp: {timestamp_str}")
            return np.nan
    
    # Try to parse as float directly
    try:
        return float(timestamp_str)
    except ValueError:
        # Try to parse as datetime string
        try:
            return pd.to_datetime(timestamp_str, errors='coerce').timestamp()
        except:
            print(f"Failed to parse timestamp: {timestamp_str}")
            return np.nan

# Convert timestamp to datetime objects
print("Parsing timestamps...")
try:
    df['timestamp_numeric'] = df[TIME_COLUMN].apply(parse_instant_timestamp)
    
    # Remove rows where timestamp parsing failed
    valid_timestamps = ~df['timestamp_numeric'].isna()
    df = df[valid_timestamps].copy()
    
    # Convert to relative time (seconds from start)
    df['time_sec'] = df['timestamp_numeric'] - df['timestamp_numeric'].min()
    
    print(f"Successfully parsed {len(df)} timestamps")
    print(f"Time range: {df['time_sec'].min():.2f}s to {df['time_sec'].max():.2f}s")

except Exception as e:
    print(f"Error parsing timestamp column '{TIME_COLUMN}': {e}")
    print("Falling back to using row index as time axis.")
    df['time_sec'] = df.index.astype(float)

# Convert BBR mode to a more readable string if it's not already
if df['bbr_mode'].dtype == 'object':
    df['bbr_mode_str'] = df['bbr_mode'].apply(lambda x: x.split('::')[-1] if isinstance(x, str) else str(x))
else:
    df['bbr_mode_str'] = df['bbr_mode'].astype(str)

# Convert boolean is_app_limited
if 'is_app_limited' in df.columns:
    if df['is_app_limited'].dtype != bool:
        df['is_app_limited'] = df['is_app_limited'].astype(str).str.lower()
        df['is_app_limited'] = df['is_app_limited'].map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)

# Convert numeric columns to proper types
numeric_columns = ['btl_bw', 'pacing_rate', 'delivery_rate_estimate', 'min_rtt_us', 
                  'latest_rtt_us', 'cwnd', 'bytes_in_flight', 'pacing_gain', 
                  'cwnd_gain', 'cycle_idx', 'newly_lost_bytes']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Data types after preprocessing:")
print(df.dtypes)

# --- Plotting Function ---
def plot_bbr_data(data, path_id_val):
    path_df = data[data[PATH_ID_COLUMN] == path_id_val].sort_values(by='time_sec')
    if path_df.empty:
        print(f"No data found for path_id {path_id_val}")
        return

    print(f"\n--- Plotting for Path ID: {path_id_val} ---")
    print(f"Number of data points: {len(path_df)}")
    print(f"Time range: {path_df['time_sec'].min():.2f}s to {path_df['time_sec'].max():.2f}s")

    fig, axs = plt.subplots(7, 1, figsize=(15, 35), sharex=True)
    time_axis = path_df['time_sec']

    # 1. BBR State
    ax = axs[0]
    # Create a mapping from state string to integer for plotting
    unique_states = path_df['bbr_mode_str'].unique()
    state_map = {state: i for i, state in enumerate(unique_states)}
    numeric_states = path_df['bbr_mode_str'].map(state_map)
    ax.plot(time_axis, numeric_states, marker='.', linestyle='-', label='BBR Mode')
    ax.set_yticks(list(state_map.values()))
    ax.set_yticklabels(list(state_map.keys()))
    ax.set_ylabel("BBR Mode")
    ax.set_title(f"BBR Internals Over Time for Path ID: {path_id_val}")
    ax.grid(True)
    ax.legend()

    # 2. Bandwidth Parameters (BtlBw, Pacing Rate, Delivery Rate)
    ax = axs[1]
    ax.plot(time_axis, path_df['btl_bw'] / 1e6, label='BtlBw (MBps)', linestyle='-')
    ax.plot(time_axis, path_df['pacing_rate'] / 1e6, label='Pacing Rate (MBps)', linestyle='--')
    ax.plot(time_axis, path_df['delivery_rate_estimate'] / 1e6, label='Delivery Rate Est. (MBps)', linestyle=':')
    ax.set_ylabel("Rate (MBps)")
    ax.grid(True)
    ax.legend()

    # 3. RTT Parameters (min_rtt, latest_rtt)
    ax = axs[2]
    ax.plot(time_axis, path_df['min_rtt_us'] / 1000, label='Min RTT (ms)')
    ax.plot(time_axis, path_df['latest_rtt_us'] / 1000, label='Latest RTT (ms)', alpha=0.7)
    ax.set_ylabel("RTT (ms)")
    ax.grid(True)
    ax.legend()

    # 4. Cwnd and Bytes in Flight
    ax = axs[3]
    ax.plot(time_axis, path_df['cwnd'] / 1024, label='Cwnd (KB)')
    ax.plot(time_axis, path_df['bytes_in_flight'] / 1024, label='Bytes in Flight (KB)', linestyle='--')
    ax.set_ylabel("Window/In-Flight (KB)")
    ax.grid(True)
    ax.legend()

    # 5. Gain Factors
    ax = axs[4]
    ax.plot(time_axis, path_df['pacing_gain'], label='Pacing Gain')
    ax.plot(time_axis, path_df['cwnd_gain'], label='Cwnd Gain', linestyle='--')
    ax.set_ylabel("Gain Factors")
    ax.grid(True)
    ax.legend()

    # 6. Loss and Application Limited
    ax = axs[5]
    ax.plot(time_axis, path_df['newly_lost_bytes'], label='Newly Lost Bytes', color='red')
    ax.set_ylabel("Lost Bytes")
    ax.tick_params(axis='y', labelcolor='red')
    ax.grid(True)
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    if 'is_app_limited' in path_df.columns:
        ax2.plot(time_axis, path_df['is_app_limited'].astype(int), label='Is App Limited', color='purple', linestyle=':')
        ax2.set_ylabel("App Limited (0=No, 1=Yes)")
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.legend(loc='upper right')
    else:
        print("Warning: 'is_app_limited' column not found.")

    # 7. ProbeBW Cycle Index (if relevant)
    # ax = axs[6]
    # ax.plot(time_axis, path_df['cycle_idx'], label='ProbeBW Cycle Index', marker='.', linestyle='None')
    # ax.set_ylabel("ProbeBW Cycle Index")
    # ax.set_xlabel("Time (seconds from start)")
    # ax.grid(True)
    # ax.legend()

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # fig.suptitle(f"BBR Internals for Path ID: {path_id_val}", fontsize=16)
    # plt.savefig(f"{OUTPUT_DIR}bbr_internals_path_{path_id_val}.png", dpi=300, bbox_inches='tight')
    # print(f"Saved plot to {OUTPUT_DIR}bbr_internals_path_{path_id_val}.png")
    # plt.close(fig)

# --- Main Execution ---
if df.empty:
    print("DataFrame is empty after loading. No plots will be generated.")
else:
    # Get unique path IDs and plot for each
    unique_path_ids = df[PATH_ID_COLUMN].unique()
    print(f"Found Path IDs: {unique_path_ids}")
    for pid in unique_path_ids:
        plot_bbr_data(df.copy(), pid)

print("\nAnalysis script finished.")