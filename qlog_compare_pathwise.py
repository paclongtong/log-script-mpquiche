import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg') # FIX: Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

PACKET_LOSS = True
CUBIC_STATES = False
PACKET_LOSS = False
CUBIC_STATES = True
class QlogConfig:
    """Configuration for a single qlog file"""
    def __init__(self, path: str, label: str, color: str = None, linestyle: str = '-'):
        self.path = path
        self.label = label
        self.color = color
        self.linestyle = linestyle

class QlogAnalyzer:
    """Analyze and compare multiple QUIC qlogs with flexible path selection"""
    
    def __init__(self, condition: str = "No Loss", paths_to_plot: List[int] = None):
        self.condition = condition
        self.paths_to_plot = paths_to_plot or [0, 1]  # Default to both paths
        self.zero_time = None
        
        # Default colors and linestyles for up to 8 qlogs
        self.default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.default_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        # CUBIC state colors and styles
        self.cubic_state_colors = {
            'SlowStart': 'green',
            'ConservativeSlowStart': 'orange',
            'CongestionAvoidance': 'blue',
            'Recovery': 'red'
        }
        
        self.cubic_state_markers = {
            'SlowStart': 'o',
            'ConservativeSlowStart': 's',
            'CongestionAvoidance': '^',
            'Recovery': 'v'
        }

    def parse_cubic_states_chunked(self, path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Parse connectivity:cubic_state_update events in chunks to handle large files efficiently.
        """
        records = []
        chunk_count = 0
        
        print(f"  Parsing CUBIC states from {Path(path).name}...")
        
        with open(path, 'r') as f:
            current_chunk = []
            
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"    Processed {line_num:,} lines...")
                
                line = line.strip()
                if not line.startswith('{'):
                    continue
                
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if event.get('name') == 'connectivity:cubic_state_update':
                    t = event.get('time')
                    if t is None:
                        continue
                    
                    data = event.get('data', {})
                    path_id = data.get('path_id', 0)
                    
                    # Only process paths we're interested in
                    if path_id not in self.paths_to_plot:
                        continue
                    
                    old_state = data.get('old_state', 'unknown')
                    new_state = data.get('new_state', 'unknown')
                    trigger = data.get('trigger', 'unknown')
                    
                    # Extract additional data if available
                    state_data = data.get('data', {})
                    cwnd = state_data.get('cwnd', 0)
                    ssthresh = state_data.get('ssthresh', 0)
                    
                    current_chunk.append({
                        'time': t,
                        'path_id': path_id,
                        'old_state': old_state,
                        'new_state': new_state,
                        'trigger': trigger,
                        'cwnd': cwnd,
                        'ssthresh': ssthresh
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        records.extend(current_chunk)
                        current_chunk = []
                        chunk_count += 1
                        
                        if chunk_count % 10 == 0:
                            gc.collect()
            
            if current_chunk:
                records.extend(current_chunk)
        
        df = pd.DataFrame(records) if records else pd.DataFrame()
        if not df.empty:
            df = df.sort_values('time').reset_index(drop=True)
        
        print(f"    Found {len(records)} CUBIC state updates")
        return df

    def parse_packet_lost_chunked(self, path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Parse recovery:packet_lost events in chunks to handle large files efficiently.
        """
        records = []
        chunk_count = 0
        
        print(f"  Parsing packet_lost from {Path(path).name}...")
        
        with open(path, 'r') as f:
            current_chunk = []
            
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"    Processed {line_num:,} lines...")
                
                line = line.strip()
                if not line.startswith('{'):
                    continue
                
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if event.get('name') == 'recovery:packet_lost':
                    t = event.get('time')
                    if t is None:
                        continue
                    
                    data = event.get('data', {})
                    path_id = data.get('path_id', 0)
                    
                    # Only process paths we're interested in
                    if path_id not in self.paths_to_plot:
                        continue
                    
                    header = data.get('header', {})
                    packet_number = header.get('packet_number', 0)
                    packet_length = header.get('length', 0)
                    trigger = data.get('trigger', 'unknown')
                    
                    current_chunk.append({
                        'time': t,
                        'path_id': path_id,
                        'packet_number': packet_number,
                        'packet_length': packet_length,
                        'trigger': trigger
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        records.extend(current_chunk)
                        current_chunk = []
                        chunk_count += 1
                        
                        if chunk_count % 10 == 0:
                            gc.collect()
            
            if current_chunk:
                records.extend(current_chunk)
        
        df = pd.DataFrame(records) if records else pd.DataFrame()
        if not df.empty:
            df = df.sort_values('time').reset_index(drop=True)
        
        print(f"    Found {len(records)} packet_lost events")
        return df
    
    def parse_metrics_qlog_chunked(self, path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Parse recovery:metrics_updated events in chunks to handle large files efficiently.
        """
        records = []
        chunk_count = 0
        
        print(f"  Parsing metrics from {Path(path).name}...")
        
        with open(path, 'r') as f:
            current_chunk = []
            
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"    Processed {line_num:,} lines...")
                
                line = line.strip()
                if not line.startswith('{'):
                    continue
                
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if event.get('name') == 'recovery:metrics_updated':
                    data = event.get('data', {})
                    t = event.get('time')
                    if t is None:
                        continue
                    
                    path_id = data.get('path_id', 0)
                    
                    # Only process paths we're interested in
                    if path_id not in self.paths_to_plot:
                        continue
                    
                    rec = {'time': t, 'path_id': path_id}
                    for field in ['bytes_in_flight', 'congestion_window',
                                  'latest_rtt', 'rtt_variance', 'pacing_rate']:
                        if field in data:
                            rec[field] = data[field]
                    
                    current_chunk.append(rec)
                    
                    if len(current_chunk) >= chunk_size:
                        records.extend(current_chunk)
                        current_chunk = []
                        chunk_count += 1
                        
                        # Periodic garbage collection for large files
                        if chunk_count % 10 == 0:
                            gc.collect()
            
            # Add remaining records
            if current_chunk:
                records.extend(current_chunk)
        
        print(f"    Found {len(records)} relevant metrics events")
        return pd.DataFrame(records) if records else pd.DataFrame()

    def parse_packet_sent_chunked(self, path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Parse transport:packet_sent events in chunks to handle large files efficiently.
        """
        records = []
        chunk_count = 0
        
        print(f"  Parsing packet_sent from {Path(path).name}...")
        
        with open(path, 'r') as f:
            current_chunk = []
            
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"    Processed {line_num:,} lines...")
                
                line = line.strip()
                if not line.startswith('{'):
                    continue
                
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if event.get('name') == 'transport:packet_sent':
                    t = event.get('time')
                    if t is None:
                        continue
                    
                    data = event.get('data', {})
                    path_id = data.get('path_id', 0)
                    
                    # Only process paths we're interested in
                    if path_id not in self.paths_to_plot:
                        continue
                    
                    raw = data.get('raw', {})
                    pl = raw.get('payload_length', 0)
                    
                    current_chunk.append({
                        'time': t, 
                        'payload_length': pl, 
                        'path_id': path_id
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        records.extend(current_chunk)
                        current_chunk = []
                        chunk_count += 1
                        
                        if chunk_count % 10 == 0:
                            gc.collect()
            
            if current_chunk:
                records.extend(current_chunk)
        
        df = pd.DataFrame(records) if records else pd.DataFrame()
        if not df.empty:
            df = df.sort_values('time').reset_index(drop=True)
        
        print(f"    Found {len(records)} relevant packet_sent events")
        return df

    def fill_missing_values(self, df: pd.DataFrame, time_col: str = 'time_ms', 
                           value_cols: List[str] = None) -> pd.DataFrame:
        """Fill missing values with forward fill to ensure continuous plots"""
        if df.empty:
            return df
        
        if value_cols is None:
            value_cols = ['congestion_window', 'latest_rtt', 'bytes_in_flight', 'pacing_rate']
        
        df_filled = df.copy()
        # Sort by time to ensure proper forward fill
        df_filled = df_filled.sort_values(time_col)
        
        for col in value_cols:
            if col in df_filled.columns:
                # Forward fill to maintain last known value
                df_filled[col] = df_filled[col].ffill()
        
        return df_filled

    def align_times(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Align all DataFrames to a common zero time"""
        if self.zero_time is None:
            # Find the earliest time across all DataFrames
            zero_times = []
            for df in dfs:
                if not df.empty and 'time' in df.columns:
                    zero_times.append(df['time'].min())
            
            if not zero_times:
                # If no data is found at all, we can't set a zero time.
                # Let's return the original list.
                return dfs
            
            self.zero_time = min(zero_times)
            print(f"Using zero time: {self.zero_time}")
        
        results = []
        for df in dfs:
            d = df.copy()
            if 'time' in d.columns and not d.empty:
                d['time_ms'] = (d['time'] - self.zero_time)
                d.sort_values('time_ms', inplace=True)
            else:
                d['time_ms'] = pd.Series([], dtype=float)
            results.append(d.reset_index(drop=True))
        
        return results

    def separate_paths(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Separate DataFrame by path_id"""
        if df.empty:
            return {path_id: pd.DataFrame() for path_id in self.paths_to_plot}
        
        path_data = {}
        for path_id in self.paths_to_plot:
            if 'path_id' in df.columns:
                path_data[path_id] = df[df['path_id'] == path_id].copy()
            else:
                # If no path_id column, assume it's path 0 data
                path_data[path_id] = df.copy() if path_id == 0 else pd.DataFrame()
        
        return path_data

    def compute_utilization_metrics(self, metrics_data: pd.DataFrame, 
                                  packet_data: pd.DataFrame, 
                                  impl_label: str, path_id: int):
        """Compute and print utilization metrics for a given path"""
        if metrics_data.empty:
            print(f"{impl_label} Path{path_id}: No data")
            return
        
        if 'congestion_window' in metrics_data.columns:
            tw = metrics_data[~metrics_data['congestion_window'].isna()].copy()
            if not tw.empty:
                times = tw['time_ms'].values
                cwnds = tw['congestion_window'].values
                if len(times) > 1:
                    deltas = (times[1:] - times[:-1]) / 1000.0
                    cwnd_area = np.sum(cwnds[:-1] * deltas)
                    max_cwnd = np.max(cwnds)
                    min_cwnd = np.min(cwnds)
                    avg_cwnd = np.mean(cwnds)
                else:
                    cwnd_area = 0.0
                    max_cwnd = cwnds[0] if len(cwnds) > 0 else 0
                    min_cwnd = max_cwnd
                    avg_cwnd = max_cwnd
            else:
                cwnd_area = max_cwnd = min_cwnd = avg_cwnd = 0
        else:
            cwnd_area = max_cwnd = min_cwnd = avg_cwnd = float('nan')
        
        total_bytes = packet_data['payload_length'].sum() if not packet_data.empty else 0
        U = total_bytes / cwnd_area if cwnd_area > 0 else float('nan')
        
        print(f"{impl_label} Path{path_id}:")
        print(f"  CWND-area = {cwnd_area:.2f} byte-seconds")
        print(f"  Total bytes sent = {total_bytes}")
        print(f"  Utilization ratio = {U:.4f}")
        print(f"  CWND stats: min={min_cwnd:.0f}, avg={avg_cwnd:.0f}, max={max_cwnd:.0f}")
        print()

    def add_packet_loss_markers(self, ax, loss_data_by_path: List[Dict], metrics_data_by_path: List[Dict], configs: List[QlogConfig]):
        """Add packet loss markers directly on the CWND curve."""
        for i, config in enumerate(configs):
            if i >= len(loss_data_by_path) or i >= len(metrics_data_by_path): continue
            
            color = config.color or self.default_colors[i % len(self.default_colors)]
            
            for path_id in self.paths_to_plot:
                loss_df = loss_data_by_path[i].get(path_id, pd.DataFrame())
                metrics_df = metrics_data_by_path[i].get(path_id, pd.DataFrame())

                if not loss_df.empty and not metrics_df.empty and 'congestion_window' in metrics_df.columns:
                    # Ensure data is sorted by time for merging
                    loss_df = loss_df.sort_values('time_ms')
                    metrics_df = metrics_df.sort_values('time_ms')
                    
                    # KEY CHANGE: Find the CWND value at the time of loss.
                    # We merge the loss events with the metrics, finding the last metric update
                    # at or before the time of the loss event ('backward').
                    merged_df = pd.merge_asof(loss_df,
                                            metrics_df[['time_ms', 'congestion_window']].dropna(),
                                            on='time_ms',
                                            direction='backward')
                    
                    if not merged_df.empty and 'congestion_window' in merged_df.columns:
                        # Drop any events where a CWND value could not be found
                        merged_df.dropna(subset=['congestion_window'], inplace=True)

                        marker = 'x' if path_id == 0 else '+'
                        
                        ax.scatter(merged_df['time_ms'], merged_df['congestion_window'],
                                    marker=marker,
                                    s=100,  # Increased size for visibility
                                    color=color,
                                    alpha=0.9,
                                    linewidth=2,
                                    zorder=5, # Ensure markers are on top
                                    label=f'{config.label} Path{path_id} Loss')

    def add_cubic_state_markers(self, ax, cubic_data_by_path: List[Dict], configs: List[QlogConfig]):
        """Add CUBIC state transition markers directly on the CWND curve."""
        for i, config in enumerate(configs):
            if i >= len(cubic_data_by_path): continue

            path_data = cubic_data_by_path[i]
            for path_id in self.paths_to_plot:
                cubic_df = path_data.get(path_id, pd.DataFrame())
                if not cubic_df.empty and 'cwnd' in cubic_df.columns:
                    
                    # Group by new_state for different marker styles
                    for state in cubic_df['new_state'].unique():
                        state_df = cubic_df[cubic_df['new_state'] == state]
                        if not state_df.empty:
                            marker = self.cubic_state_markers.get(state, 'o')
                            color = self.cubic_state_colors.get(state, 'black')
                            
                            # Get x and y coordinates from the data
                            x_values = state_df['time_ms']
                            # KEY CHANGE: Use the CWND value from the event for the y-coordinate
                            y_values = state_df['cwnd']

                            ax.scatter(x_values, y_values,
                                        marker=marker,
                                        s=80,  # Increased size for better visibility
                                        color=color,
                                        alpha=0.9,
                                        edgecolors='black',
                                        linewidth=1,
                                        zorder=5, # Ensure markers are drawn on top
                                        label=f'{config.label} Path{path_id} {state}')

    def add_packet_loss_markers_rtt(self, ax, loss_data_by_path, metrics_data_by_path, configs):
        """Places packet loss markers on the RTT plot by looking up the correct RTT value."""
        for i, config in enumerate(configs):
            if i >= len(loss_data_by_path) or i >= len(metrics_data_by_path): continue
            
            color = config.color or self.default_colors[i % len(self.default_colors)]
            
            for path_id in self.paths_to_plot:
                loss_df = loss_data_by_path[i].get(path_id, pd.DataFrame())
                metrics_df = metrics_data_by_path[i].get(path_id, pd.DataFrame())

                if not loss_df.empty and not metrics_df.empty and 'latest_rtt' in metrics_df.columns:
                    # Ensure metrics have continuous RTT values through forward fill
                    metrics_filled = metrics_df.copy()
                    metrics_filled = metrics_filled.sort_values('time_ms')
                    metrics_filled['latest_rtt'] = metrics_filled['latest_rtt'].ffill()
                    
                    metric_subset = metrics_filled[['time_ms', 'latest_rtt']].dropna().sort_values('time_ms')
                    merged_df = pd.merge_asof(loss_df.sort_values('time_ms'), metric_subset, on='time_ms', direction='backward')

                    if not merged_df.empty and 'latest_rtt' in merged_df.columns:
                        merged_df.dropna(subset=['latest_rtt'], inplace=True)
                        rtt_values = merged_df['latest_rtt'].copy()
                        # Apply same conversion logic as main RTT plot
                        if not rtt_values.empty and rtt_values.max() > 1000:
                            rtt_values = rtt_values / 1000
                        y_values = rtt_values

                        marker = 'x' if path_id == 0 else '+'
                        ax.scatter(merged_df['time_ms'], y_values,
                                     marker=marker, s=100, color=color, alpha=0.9,
                                     linewidth=2, zorder=5, label=f'{config.label} Path{path_id} Loss')

    def add_cubic_state_markers_rtt(self, ax, cubic_data_by_path, metrics_data_by_path, configs):
        """Places CUBIC state markers on the RTT plot by looking up the correct RTT value."""
        for i, config in enumerate(configs):
            if i >= len(cubic_data_by_path) or i >= len(metrics_data_by_path): continue

            for path_id in self.paths_to_plot:
                cubic_df = cubic_data_by_path[i].get(path_id, pd.DataFrame())
                metrics_df = metrics_data_by_path[i].get(path_id, pd.DataFrame())

                if not cubic_df.empty and not metrics_df.empty and 'latest_rtt' in metrics_df.columns:
                    # Ensure metrics have continuous RTT values through forward fill
                    metrics_filled = metrics_df.copy()
                    metrics_filled = metrics_filled.sort_values('time_ms')
                    metrics_filled['latest_rtt'] = metrics_filled['latest_rtt'].ffill()
                    
                    metric_subset = metrics_filled[['time_ms', 'latest_rtt']].dropna().sort_values('time_ms')
                    merged_df = pd.merge_asof(cubic_df.sort_values('time_ms'), metric_subset, on='time_ms', direction='backward')

                    if not merged_df.empty and 'latest_rtt' in merged_df.columns:
                        merged_df.dropna(subset=['latest_rtt'], inplace=True)
                        for state in merged_df['new_state'].unique():
                            state_df = merged_df[merged_df['new_state'] == state]
                            if not state_df.empty:
                                rtt_values = state_df['latest_rtt'].copy()
                                # Apply same conversion logic as main RTT plot
                                if not rtt_values.empty and rtt_values.max() > 1000:
                                    rtt_values = rtt_values / 1000
                                y_values = rtt_values

                                marker = self.cubic_state_markers.get(state, 'o')
                                color = self.cubic_state_colors.get(state, 'black')
                                ax.scatter(state_df['time_ms'], y_values,
                                             marker=marker, s=80, color=color, alpha=0.9,
                                             edgecolors='black', linewidth=1, zorder=5,
                                             label=f'{config.label} Path{path_id} {state}')

    def get_common_time_range(self, all_data: List[List[Dict]]) -> Tuple[float, float]:
        """Get common time range across all datasets"""
        min_times = []
        max_times = []
        
        for data_list in all_data:
            for data_by_path in data_list:
                if data_by_path is None: continue # Safety check
                for path_data in data_by_path.values():
                    if not path_data.empty and 'time_ms' in path_data.columns:
                        min_times.append(path_data['time_ms'].min())
                        max_times.append(path_data['time_ms'].max())
        
        if min_times and max_times:
            return min(min_times), max(max_times)
        return 0, 10000

    def plot_congestion_window(self, all_metrics: List[Dict], all_losses: List[Dict], 
                                all_cubic_states: List[Dict], configs: List[QlogConfig], 
                                output_prefix: str = None):
        """Plot congestion window comparison with packet loss markers and CUBIC states."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (metrics_by_path, config) in enumerate(zip(all_metrics, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty and 'congestion_window' in df.columns:
                    linestyle = base_linestyle if path_id == 0 else '--'
                    label = f'{config.label} Path{path_id}'
                    
                    ax.plot(df['time_ms'], df['congestion_window'], 
                        label=label, color=color, linestyle=linestyle, 
                        linewidth=2, alpha=0.8)
        
        y_lim_set = ax.get_ylim()[1] > 0
        
        # Add packet loss markers
        if PACKET_LOSS and y_lim_set:
            # KEY CHANGE: Pass 'all_metrics' to the function
            self.add_packet_loss_markers(ax, all_losses, all_metrics, configs)
        
        # Add CUBIC state markers
        if CUBIC_STATES and y_lim_set:
            self.add_cubic_state_markers(ax, all_cubic_states, configs)
        
        time_min, time_max = self.get_common_time_range([all_metrics, all_losses, all_cubic_states])
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Congestion Window - {self.condition}', fontsize=16)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('CWND (bytes)', fontsize=12)
        
        # Consolidate legend to avoid duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_cwnd_comparison.png', dpi=300, bbox_inches='tight')
            print(f"CWND plot saved to {output_prefix}_cwnd_comparison.png")
        else:
            plt.show()

    def plot_bytes_in_flight(self, all_metrics: List[Dict], all_losses: List[Dict],
                           all_cubic_states: List[Dict], configs: List[QlogConfig], 
                           output_prefix: str = None):
        """Plot bytes in flight comparison with packet loss markers and CUBIC states"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (metrics_by_path, config) in enumerate(zip(all_metrics, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty and 'bytes_in_flight' in df.columns:
                    linestyle = base_linestyle if path_id == 0 else '--'
                    label = f'{config.label} Path{path_id} bytes_in_flight'
                    
                    ax.plot(df['time_ms'], df['bytes_in_flight'], 
                           label=label, color=color, linestyle=linestyle,
                           linewidth=2, alpha=0.6)
        
        y_lim_set = ax.get_ylim()[1] > 0
        
        if PACKET_LOSS and y_lim_set:
            self.add_packet_loss_markers(ax, all_losses, configs)
        
        if CUBIC_STATES and y_lim_set:
            self.add_cubic_state_markers(ax, all_cubic_states, configs)
        
        time_min, time_max = self.get_common_time_range([all_metrics, all_losses, all_cubic_states])
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Bytes In Flight - {self.condition}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Bytes', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_bytes_in_flight.png', dpi=300, bbox_inches='tight')
            print(f"Bytes in flight plot saved to {output_prefix}_bytes_in_flight.png")
        else:
            plt.show()

    def plot_rtt_comparison(self, all_metrics: List[Dict], all_losses: List[Dict],
                          all_cubic_states: List[Dict], 
                          configs: List[QlogConfig], 
                          output_prefix: str = None):
        """Plot RTT comparison with packet loss markers and CUBIC states"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (metrics_by_path, config) in enumerate(zip(all_metrics, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty and 'latest_rtt' in df.columns:
                    rtt = df['latest_rtt'].copy()
                    # Convert RTT to milliseconds if it's in microseconds (threshold > 1000ms)
                    # if not rtt.empty and rtt.max() > 1000:
                    #     rtt = rtt / 1000
                    
                    linestyle = base_linestyle if path_id == 0 else '--'
                    label = f'{config.label} Path{path_id}'
                    
                    ax.plot(df['time_ms'], rtt, 
                           label=label, color=color, linestyle=linestyle,
                           linewidth=2, alpha=0.8)
        
        y_lim_set = ax.get_ylim()[1] > 0
        
        if PACKET_LOSS and y_lim_set:
            self.add_packet_loss_markers_rtt(ax, all_losses, all_metrics, configs)
        
        if CUBIC_STATES and y_lim_set:
            self.add_cubic_state_markers_rtt(ax, all_cubic_states, all_metrics, configs)
        
        time_min, time_max = self.get_common_time_range([all_metrics, all_losses, all_cubic_states])
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Round-Trip Time - {self.condition}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('RTT (ms)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_rtt_comparison.png', dpi=300, bbox_inches='tight')
            print(f"RTT plot saved to {output_prefix}_rtt_comparison.png")
        else:
            plt.show()

    def plot_rtt_raw_updates(self, all_metrics: List[Dict], all_losses: List[Dict],
                            all_cubic_states: List[Dict], 
                            configs: List[QlogConfig], 
                            output_prefix: str = None):
        """Plot RTT without forward filling to show actual RTT update occurrences"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create copies of metrics data without forward filling applied
        all_metrics_raw = []
        for i, metrics_by_path in enumerate(all_metrics):
            metrics_raw_by_path = {}
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty:
                    # Only keep rows that actually have RTT values (no forward fill)
                    df_raw = df.copy()
                    if 'latest_rtt' in df_raw.columns:
                        # Keep only non-null RTT values to show actual updates
                        df_raw = df_raw.dropna(subset=['latest_rtt'])
                    metrics_raw_by_path[path_id] = df_raw
                else:
                    metrics_raw_by_path[path_id] = pd.DataFrame()
            all_metrics_raw.append(metrics_raw_by_path)
        
        for i, (metrics_by_path, config) in enumerate(zip(all_metrics_raw, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty and 'latest_rtt' in df.columns:
                    rtt = df['latest_rtt'].copy()
                    # Convert RTT to milliseconds if it's in microseconds (threshold > 1000ms)
                    # if not rtt.empty and rtt.max() > 1000:
                    #     rtt = rtt / 1000
                    
                    linestyle = base_linestyle if path_id == 0 else '--'
                    label = f'{config.label} Path{path_id} (Raw Updates)'
                    
                    # Use scatter plot to show individual RTT measurements
                    ax.scatter(df['time_ms'], rtt, 
                             label=label, color=color, 
                             s=20, alpha=0.7, marker='o')
                    
                    # Optionally add a line connecting the points
                    ax.plot(df['time_ms'], rtt, 
                           color=color, linestyle=linestyle,
                           linewidth=1, alpha=0.5)
        
        y_lim_set = ax.get_ylim()[1] > 0
        
        # Add packet loss markers using raw metrics data
        if PACKET_LOSS and y_lim_set:
            # Create marker functions that work with raw data
            self.add_packet_loss_markers_rtt_raw(ax, all_losses, all_metrics_raw, configs)
        
        if CUBIC_STATES and y_lim_set:
            self.add_cubic_state_markers_rtt_raw(ax, all_cubic_states, all_metrics_raw, configs)
        
        time_min, time_max = self.get_common_time_range([all_metrics_raw, all_losses, all_cubic_states])
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Round-Trip Time (Raw Updates) - {self.condition}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('RTT (ms)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_rtt_raw_updates.png', dpi=300, bbox_inches='tight')
            print(f"RTT raw updates plot saved to {output_prefix}_rtt_raw_updates.png")
        else:
            plt.show()

    def add_packet_loss_markers_rtt_raw(self, ax, loss_data_by_path, metrics_data_by_path, configs):
        """Places packet loss markers on the raw RTT plot by looking up the nearest RTT value."""
        for i, config in enumerate(configs):
            if i >= len(loss_data_by_path) or i >= len(metrics_data_by_path): continue
            
            color = config.color or self.default_colors[i % len(self.default_colors)]
            
            for path_id in self.paths_to_plot:
                loss_df = loss_data_by_path[i].get(path_id, pd.DataFrame())
                metrics_df = metrics_data_by_path[i].get(path_id, pd.DataFrame())

                if not loss_df.empty and not metrics_df.empty and 'latest_rtt' in metrics_df.columns:
                    # Use raw RTT data (no forward fill)
                    metric_subset = metrics_df[['time_ms', 'latest_rtt']].dropna().sort_values('time_ms')
                    merged_df = pd.merge_asof(loss_df.sort_values('time_ms'), metric_subset, on='time_ms', direction='backward')

                    if not merged_df.empty and 'latest_rtt' in merged_df.columns:
                        merged_df.dropna(subset=['latest_rtt'], inplace=True)
                        rtt_values = merged_df['latest_rtt'].copy()
                        # Apply same conversion logic as main RTT plot
                        if not rtt_values.empty and rtt_values.max() > 1000:
                            rtt_values = rtt_values / 1000
                        y_values = rtt_values

                        marker = 'x' if path_id == 0 else '+'
                        ax.scatter(merged_df['time_ms'], y_values,
                                     marker=marker, s=120, color=color, alpha=0.9,
                                     linewidth=2, zorder=6, label=f'{config.label} Path{path_id} Loss (Raw)')

    def add_cubic_state_markers_rtt_raw(self, ax, cubic_data_by_path, metrics_data_by_path, configs):
        """Places CUBIC state markers on the raw RTT plot by looking up the nearest RTT value."""
        for i, config in enumerate(configs):
            if i >= len(cubic_data_by_path) or i >= len(metrics_data_by_path): continue

            for path_id in self.paths_to_plot:
                cubic_df = cubic_data_by_path[i].get(path_id, pd.DataFrame())
                metrics_df = metrics_data_by_path[i].get(path_id, pd.DataFrame())

                if not cubic_df.empty and not metrics_df.empty and 'latest_rtt' in metrics_df.columns:
                    # Use raw RTT data (no forward fill)
                    metric_subset = metrics_df[['time_ms', 'latest_rtt']].dropna().sort_values('time_ms')
                    merged_df = pd.merge_asof(cubic_df.sort_values('time_ms'), metric_subset, on='time_ms', direction='backward')

                    if not merged_df.empty and 'latest_rtt' in merged_df.columns:
                        merged_df.dropna(subset=['latest_rtt'], inplace=True)
                        for state in merged_df['new_state'].unique():
                            state_df = merged_df[merged_df['new_state'] == state]
                            if not state_df.empty:
                                rtt_values = state_df['latest_rtt'].copy()
                                # Apply same conversion logic as main RTT plot
                                if not rtt_values.empty and rtt_values.max() > 1000:
                                    rtt_values = rtt_values / 1000
                                y_values = rtt_values

                                marker = self.cubic_state_markers.get(state, 'o')
                                color = self.cubic_state_colors.get(state, 'black')
                                ax.scatter(state_df['time_ms'], y_values,
                                             marker=marker, s=100, color=color, alpha=0.9,
                                             edgecolors='black', linewidth=1, zorder=6,
                                             label=f'{config.label} Path{path_id} {state} (Raw)')

    def plot_send_rate(self, all_packets: List[Dict], all_losses: List[Dict],
                      all_cubic_states: List[Dict], configs: List[QlogConfig], 
                      output_prefix: str = None, window_ms: int = 100):
        """Plot send rate comparison with packet loss markers and CUBIC states"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (packets_by_path, config) in enumerate(zip(all_packets, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df_send = packets_by_path.get(path_id, pd.DataFrame())
                if not df_send.empty and 'time_ms' in df_send.columns:
                    max_time = df_send['time_ms'].max()
                    if max_time > 0:
                        time_bins = np.arange(0, max_time + window_ms, window_ms)
                        send_rates = []
                        bin_centers = []
                        
                        for j in range(len(time_bins) - 1):
                            mask = ((df_send['time_ms'] >= time_bins[j]) & 
                                   (df_send['time_ms'] < time_bins[j+1]))
                            bytes_in_bin = df_send.loc[mask, 'payload_length'].sum()
                            send_rate_mbps = (bytes_in_bin * 8) / (window_ms / 1000) / 1e6
                            send_rates.append(send_rate_mbps)
                            bin_centers.append((time_bins[j] + time_bins[j+1]) / 2)
                        
                        if bin_centers:
                            linestyle = base_linestyle if path_id == 0 else '--'
                            label = f'{config.label} Path{path_id}'
                            
                            ax.plot(bin_centers, send_rates, label=label, 
                                   color=color, linestyle=linestyle,
                                   linewidth=2, alpha=0.8)
        
        y_lim_set = ax.get_ylim()[1] > 0
        
        if PACKET_LOSS and y_lim_set:
            self.add_packet_loss_markers(ax, all_losses, configs)
        
        if CUBIC_STATES and y_lim_set:
            self.add_cubic_state_markers(ax, all_cubic_states, configs)
        
        time_min, time_max = self.get_common_time_range([all_packets, all_losses, all_cubic_states])
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Send Rate (Mbps) - {self.condition}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Rate (Mbps)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_send_rate.png', dpi=300, bbox_inches='tight')
            print(f"Send rate plot saved to {output_prefix}_send_rate.png")
        else:
            plt.show()

    def print_packet_loss_summary(self, all_losses: List[Dict], configs: List[QlogConfig]):
        """Print summary of packet loss events"""
        print(f"\n=== Packet Loss Summary ===")
        for i, config in enumerate(configs):
            if i >= len(all_losses): continue
            
            total_losses = 0
            path_data = all_losses[i]
            for path_id in self.paths_to_plot:
                loss_df = path_data.get(path_id, pd.DataFrame())
                path_losses = len(loss_df)
                total_losses += path_losses
                if path_losses > 0:
                    triggers = loss_df['trigger'].value_counts()
                    print(f"{config.label} Path{path_id}: {path_losses} losses")
                    for trigger, count in triggers.items():
                        print(f"  {trigger}: {count}")
            
            if total_losses == 0:
                print(f"{config.label}: No packet losses detected")
        print()

    def analyze(self, configs: List[QlogConfig], output_prefix: str = None):
        """Main analysis function"""
        print(f"Starting analysis of {len(configs)} qlog files...")
        print(f"Analyzing paths: {self.paths_to_plot}")
        print(f"Condition: {self.condition}")
        print()
        
        all_metrics_raw = []
        all_packets_raw = []
        all_losses_raw = []
        all_cubic_states_raw = []
        
        # Parse all files first
        for i, config in enumerate(configs):
            print(f"Processing {config.label} ({i+1}/{len(configs)}):")
            
            all_metrics_raw.append(self.parse_metrics_qlog_chunked(config.path))
            all_packets_raw.append(self.parse_packet_sent_chunked(config.path))
            all_losses_raw.append(self.parse_packet_lost_chunked(config.path))
            if CUBIC_STATES:
                all_cubic_states_raw.append(self.parse_cubic_states_chunked(config.path))
            
            gc.collect()
        
        # Align timelines based on all collected data
        print("\nAligning timelines...")
        all_combined_raw = all_metrics_raw + all_packets_raw + all_losses_raw + all_cubic_states_raw
        all_combined_aligned = self.align_times(all_combined_raw)
        
        # FIX: Complete and correct the logic to split aligned data back
        n_files = len(configs)
        all_metrics_raw = all_combined_aligned[0 : n_files]
        all_packets_raw = all_combined_aligned[n_files : 2*n_files]
        all_losses_raw = all_combined_aligned[2*n_files : 3*n_files]
        if CUBIC_STATES:
            all_cubic_states_raw = all_combined_aligned[3*n_files : 4*n_files]

        # Separate data by path_id
        all_metrics = [self.separate_paths(df) for df in all_metrics_raw]
        all_packets = [self.separate_paths(df) for df in all_packets_raw]
        all_losses = [self.separate_paths(df) for df in all_losses_raw]
        
        # FIX: Initialize all_cubic_states as an empty list to prevent errors.
        all_cubic_states = []
        if CUBIC_STATES:
            all_cubic_states = [self.separate_paths(df) for df in all_cubic_states_raw]
        
        # Forward fill missing values for each path in metrics data to ensure continuous plots
        print("Forward filling missing values for continuous plotting...")
        for i in range(len(all_metrics)):
            for path_id in self.paths_to_plot:
                if path_id in all_metrics[i] and not all_metrics[i][path_id].empty:
                    all_metrics[i][path_id] = self.fill_missing_values(all_metrics[i][path_id])

        # Print summaries before plotting
        if PACKET_LOSS:
            self.print_packet_loss_summary(all_losses, configs)

        # Generate all plots
        print("\nGenerating plots...")
        self.plot_congestion_window(all_metrics, all_losses, all_cubic_states, configs, output_prefix)
        self.plot_bytes_in_flight(all_metrics, all_losses, all_cubic_states, configs, output_prefix)
        self.plot_rtt_comparison(all_metrics, all_losses, all_cubic_states, configs, output_prefix)
        self.plot_rtt_raw_updates(all_metrics, all_losses, all_cubic_states, configs, output_prefix)
        # self.plot_send_rate(all_packets, all_losses, all_cubic_states, configs, output_prefix)
        
        print("\nAnalysis complete.")
        plt.close('all') # Close all figures to free memory
        gc.collect()


def parse_qlog_configs(qlog_args: List[str]) -> List[QlogConfig]:
    """Parse qlog configuration from command line arguments"""
    configs = []
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    default_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    for i, qlog_spec in enumerate(qlog_args):
        parts = qlog_spec.split(',')
        if len(parts) < 2:
            raise ValueError(f"Invalid qlog specification: {qlog_spec}. Expected format: path,label[,color[,linestyle]]")
        
        path = parts[0].strip()
        label = parts[1].strip()
        color = parts[2].strip() if len(parts) > 2 and parts[2].strip() else default_colors[i % len(default_colors)]
        # linestyle = parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_linestyles[i % len(default_linestyles)]
        linestyle = parts[3].strip() if len(parts) > 3 and parts[3].strip() else '-'
        
        configs.append(QlogConfig(path, label, color, linestyle))
    
    return configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Multi-Path CWND between multiple QUIC implementations with packet loss tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 2 qlogs, both paths, with packet loss tracking
  python script.py --qlogs "client1.qlog,Client 1" "server1.qlog,Server 1"
  
  # Compare 3 qlogs, only path 0, with custom colors
  python script.py --qlogs "c1.qlog,Client 1,blue" "c2.qlog,Client 2,red,--" "s1.qlog,Server,green" --paths 0
  
  # Compare multiple qlogs with condition and packet loss visualization
  python script.py --qlogs "loss1.qlog,2% Loss" "noloss.qlog,No Loss" --condition "Network Comparison" --paths 0 1
        """)
    
    parser.add_argument('--qlogs', nargs='+', required=True,
                        help='Qlog specifications in format: "path,label[,color[,linestyle]]"')
    parser.add_argument('--paths', nargs='+', type=int, default=[0, 1],
                        help='Path IDs to analyze (default: 0 1)')
    parser.add_argument('--condition', default='No Loss',
                        help='Condition description for plot titles')
    parser.add_argument('--output-prefix', 
                        help='Prefix for output files (will create multiple plots)')
    
    args = parser.parse_args()
    
    try:
        # Parse qlog configurations
        configs = parse_qlog_configs(args.qlogs)
        
        # Validate paths
        valid_paths = [p for p in args.paths if p >= 0]
        if not valid_paths:
            raise ValueError("At least one valid path ID (>= 0) must be specified")
        
        # Create analyzer and run analysis
        analyzer = QlogAnalyzer(condition=args.condition, paths_to_plot=valid_paths)
        analyzer.analyze(configs, args.output_prefix)
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)