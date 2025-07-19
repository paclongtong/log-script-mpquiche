import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import defaultdict

def parse_quic_log(file_path):
    """Parse QUIC log file and extract ping frame information"""
    ping_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON log entry
                log_entry = json.loads(line)
                
                # Check if this is a packet sent/received event with ping frames
                if (log_entry.get('name') in ['transport:packet_sent', 'transport:packet_received'] and 
                    'data' in log_entry and 'frames' in log_entry['data']):
                    
                    frames = log_entry['data']['frames']
                    
                    # Look for ping frames
                    for frame in frames:
                        if frame.get('frame_type') == 'ping':
                            ping_data.append({
                                'time': log_entry['time'],
                                'event_type': log_entry['name'],
                                'path_id': log_entry['data'].get('path_id', 0),
                                'packet_number': log_entry['data']['header'].get('packet_number', 0),
                                'packet_type': log_entry['data']['header'].get('packet_type', ''),
                                'payload_length': log_entry['data']['raw'].get('payload_length', 0)
                            })
                            
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue
    
    return pd.DataFrame(ping_data)

def analyze_ping_frames(df):
    """Analyze ping frame patterns"""
    if df.empty:
        print("No ping frames found in the log")
        return
    
    # Basic statistics
    print("=== PING FRAME ANALYSIS ===")
    print(f"Total ping frames: {len(df)}")
    print(f"Time range: {df['time'].min():.3f} - {df['time'].max():.3f} seconds")
    print(f"Duration: {df['time'].max() - df['time'].min():.3f} seconds")
    print(f"Unique paths: {df['path_id'].nunique()}")
    print(f"Path distribution: {df['path_id'].value_counts().to_dict()}")
    
    # Calculate intervals between ping frames
    df_sorted = df.sort_values('time')
    df_sorted['time_diff'] = df_sorted['time'].diff()
    
    print(f"\nPing interval statistics:")
    print(f"Mean interval: {df_sorted['time_diff'].mean():.3f} seconds")
    print(f"Median interval: {df_sorted['time_diff'].median():.3f} seconds")
    print(f"Min interval: {df_sorted['time_diff'].min():.3f} seconds")
    print(f"Max interval: {df_sorted['time_diff'].max():.3f} seconds")
    
    return df_sorted

def create_visualizations(df):
    """Create comprehensive visualizations of ping frame data"""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Timeline plot - Ping frames over time by path
    ax1 = plt.subplot(3, 2, 1)
    for path_id in sorted(df['path_id'].unique()):
        path_data = df[df['path_id'] == path_id]
        plt.scatter(path_data['time'], [path_id] * len(path_data), 
                   alpha=0.6, s=20, label=f'Path {path_id}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Path ID')
    plt.title('Ping Frames Timeline by Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Ping frequency over time (sliding window)
    ax2 = plt.subplot(3, 2, 2)
    window_size = 1000  # 1 second window
    
    # Create time bins
    time_min, time_max = df['time'].min(), df['time'].max()
    time_bins = np.arange(time_min, time_max + window_size, window_size)
    
    for path_id in sorted(df['path_id'].unique()):
        path_data = df[df['path_id'] == path_id]
        counts, bins = np.histogram(path_data['time'], bins=time_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, counts, marker='o', markersize=3, 
                label=f'Path {path_id}', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Ping Count per {window_size}ms')
    plt.title('Ping Frequency Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution of ping intervals
    ax3 = plt.subplot(3, 2, 3)
    df_sorted = df.sort_values('time')
    
    for path_id in sorted(df['path_id'].unique()):
        path_data = df_sorted[df_sorted['path_id'] == path_id].copy()
        if len(path_data) > 1:
            intervals = path_data['time'].diff().dropna()
            plt.hist(intervals, bins=30, alpha=0.6, 
                    label=f'Path {path_id}', density=True)
    
    plt.xlabel('Interval (seconds)')
    plt.ylabel('Density')
    plt.title('Distribution of Ping Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative ping count
    ax4 = plt.subplot(3, 2, 4)
    for path_id in sorted(df['path_id'].unique()):
        path_data = df[df['path_id'] == path_id].sort_values('time')
        cumulative = np.arange(1, len(path_data) + 1)
        plt.plot(path_data['time'], cumulative, 
                label=f'Path {path_id}', linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Ping Count')
    plt.title('Cumulative Ping Frames by Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Ping rate (pings per second) over time
    ax5 = plt.subplot(3, 2, 5)
    window_size = 5000  # 5 second sliding window
    
    for path_id in sorted(df['path_id'].unique()):
        path_data = df[df['path_id'] == path_id].sort_values('time')
        if len(path_data) > 1:
            # Calculate sliding window ping rate
            times = path_data['time'].values
            rates = []
            rate_times = []
            
            for i in range(len(times)):
                window_start = times[i] - window_size
                window_end = times[i]
                pings_in_window = np.sum((times >= window_start) & (times <= window_end))
                rate = pings_in_window / (window_size / 1000)  # pings per second
                rates.append(rate)
                rate_times.append(times[i])
            
            plt.plot(rate_times, rates, label=f'Path {path_id}', 
                    alpha=0.7, linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Ping Rate (pings/second)')
    plt.title(f'Ping Rate Over Time ({window_size}ms window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Event type distribution
    ax6 = plt.subplot(3, 2, 6)
    event_counts = df.groupby(['path_id', 'event_type']).size().unstack(fill_value=0)
    event_counts.plot(kind='bar', ax=ax6, width=0.8)
    plt.xlabel('Path ID')
    plt.ylabel('Count')
    plt.title('Ping Frame Events by Type and Path')
    plt.legend(title='Event Type')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()
    
    # Additional detailed statistics
    print("\n=== DETAILED PATH STATISTICS ===")
    for path_id in sorted(df['path_id'].unique()):
        path_data = df[df['path_id'] == path_id].sort_values('time')
        print(f"\nPath {path_id}:")
        print(f"  Total pings: {len(path_data)}")
        print(f"  Time range: {path_data['time'].min():.3f} - {path_data['time'].max():.3f}")
        print(f"  Duration: {path_data['time'].max() - path_data['time'].min():.3f} seconds")
        
        if len(path_data) > 1:
            intervals = path_data['time'].diff().dropna()
            print(f"  Avg interval: {intervals.mean():.3f} seconds")
            print(f"  Interval std: {intervals.std():.3f} seconds")
            print(f"  Ping rate: {len(path_data) / (path_data['time'].max() - path_data['time'].min()):.3f} pings/second")

def main():
    """Main function to run the analysis"""
    # Replace with your actual file path
    log_file = "/tmp/minitopo_experiences/client-39c90057a95e2ac8991bbaf1a511b0f618934b79.sqlog"
    
    print("Parsing QUIC log file...")
    df = parse_quic_log(log_file)
    
    if df.empty:
        print("No ping frames found in the log file.")
        return
    
    print(f"Found {len(df)} ping frames")
    
    # Analyze the data
    df_analyzed = analyze_ping_frames(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df_analyzed)
    
    # Save processed data
    df_analyzed.to_csv('ping_frames_analysis.csv', index=False)
    print("\nProcessed data saved to 'ping_frames_analysis.csv'")

if __name__ == "__main__":
    main()