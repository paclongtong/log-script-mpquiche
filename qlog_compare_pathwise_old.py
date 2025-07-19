import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

PACKET_LOSS = False

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
        """Fill missing values with forward fill"""
        if df.empty:
            return df
        
        if value_cols is None:
            value_cols = ['congestion_window', 'latest_rtt']
        
        df_filled = df.copy()
        for col in value_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(method='ffill')
        
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
                raise RuntimeError('No events found to align timelines.')
            
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

    def add_packet_loss_markers(self, ax, loss_data_by_path: Dict, configs: List[QlogConfig]):
        """Add packet loss markers to a plot"""
        for i, config in enumerate(configs):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            
            for path_id in self.paths_to_plot:
                if i < len(loss_data_by_path) and path_id in loss_data_by_path[i]:
                    loss_df = loss_data_by_path[i][path_id]
                    if not loss_df.empty:
                        # Use different marker shapes for different paths
                        marker = 'x' if path_id == 0 else '+'
                        marker_size = 80 if path_id == 0 else 60
                        
                        # Get y-limits to position markers at top of plot
                        y_min, y_max = ax.get_ylim()
                        y_pos = y_max * 0.95  # Position near top of plot
                        
                        ax.scatter(loss_df['time_ms'], [y_pos] * len(loss_df),
                                 marker=marker, s=marker_size, color=color, alpha=0.8,
                                 label=f'{config.label} Path{path_id} Loss' if len(loss_df) > 0 else None)

    def get_common_time_range(self, all_data: List[Dict]) -> Tuple[float, float]:
        """Get common time range across all datasets"""
        min_times = []
        max_times = []
        
        for data_by_path in all_data:
            for path_data in data_by_path.values():
                if not path_data.empty and 'time_ms' in path_data.columns:
                    min_times.append(path_data['time_ms'].min())
                    max_times.append(path_data['time_ms'].max())
        
        if min_times and max_times:
            return min(min_times), max(max_times)
        return 0, 10000  # Default range

    def plot_congestion_window(self, all_metrics: List[Dict], all_losses: List[Dict], 
                              configs: List[QlogConfig], output_prefix: str = None):
        """Plot congestion window comparison with packet loss markers"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (metrics_by_path, config) in enumerate(zip(all_metrics, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty and 'congestion_window' in df.columns:
                    # Vary linestyle for different paths
                    linestyle = base_linestyle if path_id == 0 else '--'
                    label = f'{config.label} Path{path_id}'
                    
                    ax.plot(df['time_ms'], df['congestion_window'], 
                           label=label, color=color, linestyle=linestyle, 
                           linewidth=2, alpha=0.8)
        
        # Add packet loss markers
        if PACKET_LOSS:
            self.add_packet_loss_markers(ax, all_losses, configs)
        
        # Set common time range
        time_min, time_max = self.get_common_time_range(all_metrics + all_losses)
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Congestion Window - {self.condition}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('CWND (bytes)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_cwnd_comparison.png', dpi=300, bbox_inches='tight')
            print(f"CWND plot saved to {output_prefix}_cwnd_comparison.png")
        else:
            # plt.show()
            pass

    def plot_bytes_in_flight(self, all_metrics: List[Dict], all_losses: List[Dict],
                           configs: List[QlogConfig], output_prefix: str = None):
        """Plot bytes in flight comparison with packet loss markers"""
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
        
        # Add packet loss markers
        if PACKET_LOSS:
            self.add_packet_loss_markers(ax, all_losses, configs)
        
        # Set common time range
        time_min, time_max = self.get_common_time_range(all_metrics + all_losses)
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
            # plt.show()
            pass

    def plot_rtt_comparison(self, all_metrics: List[Dict], all_losses: List[Dict],
                          configs: List[QlogConfig], output_prefix: str = None):
        """Plot RTT comparison with packet loss markers"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (metrics_by_path, config) in enumerate(zip(all_metrics, configs)):
            color = config.color or self.default_colors[i % len(self.default_colors)]
            base_linestyle = config.linestyle
            
            for path_id in self.paths_to_plot:
                df = metrics_by_path.get(path_id, pd.DataFrame())
                if not df.empty and 'latest_rtt' in df.columns:
                    rtt = df['latest_rtt'].copy()
                    # Convert RTT to milliseconds if it's in microseconds
                    if not rtt.empty and rtt.max() > 1000:
                        rtt = rtt / 1000
                    
                    linestyle = base_linestyle if path_id == 0 else '--'
                    label = f'{config.label} Path{path_id}'
                    
                    ax.plot(df['time_ms'], rtt, 
                           label=label, color=color, linestyle=linestyle,
                           linewidth=2, alpha=0.8)
        
        # Add packet loss markers
        if PACKET_LOSS:
            self.add_packet_loss_markers(ax, all_losses, configs)
        print(f"    Found {len(df)} metrics records, RTT range: {df['latest_rtt'].min():.2f} - {df['latest_rtt'].max():.2f}")
        # Set common time range
        time_min, time_max = self.get_common_time_range(all_metrics + all_losses)
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
            # plt.show()
            pass

    def plot_send_rate(self, all_packets: List[Dict], all_losses: List[Dict],
                      configs: List[QlogConfig], output_prefix: str = None, window_ms: int = 100):
        """Plot send rate comparison with packet loss markers"""
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
                            packets_in_bin = mask.sum()
                            send_rate = packets_in_bin * (1000 / window_ms)  # packets per second
                            send_rates.append(send_rate)
                            bin_centers.append((time_bins[j] + time_bins[j+1]) / 2)
                        
                        if bin_centers:
                            linestyle = base_linestyle if path_id == 0 else '--'
                            label = f'{config.label} Path{path_id}'
                            
                            ax.plot(bin_centers, send_rates, label=label, 
                                   color=color, linestyle=linestyle,
                                   linewidth=2, alpha=0.8)
        
        # Add packet loss markers
        if PACKET_LOSS:
            self.add_packet_loss_markers(ax, all_losses, configs)
        
        # Set common time range
        time_min, time_max = self.get_common_time_range(all_packets + all_losses)
        ax.set_xlim(time_min, time_max)
        
        ax.set_title(f'Send Rate - {self.condition}', fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Packets/sec', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f'{output_prefix}_send_rate.png', dpi=300, bbox_inches='tight')
            print(f"Send rate plot saved to {output_prefix}_send_rate.png")
        else:
            # plt.show()
            pass

    def print_packet_loss_summary(self, all_losses: List[Dict], configs: List[QlogConfig]):
        """Print summary of packet loss events"""
        print(f"\n=== Packet Loss Summary ===")
        for i, config in enumerate(configs):
            if i < len(all_losses):
                total_losses = 0
                for path_id in self.paths_to_plot:
                    path_losses = len(all_losses[i].get(path_id, []))
                    total_losses += path_losses
                    if path_losses > 0:
                        loss_df = all_losses[i][path_id]
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
        
        all_metrics = []
        all_packets = []
        all_losses = []
        all_metrics_raw = []
        all_packets_raw = []
        all_losses_raw = []
        
        # Parse all files
        for i, config in enumerate(configs):
            print(f"Processing {config.label} ({i+1}/{len(configs)}):")
            
            # Parse metrics, packets, and losses
            metrics = self.parse_metrics_qlog_chunked(config.path)
            packets = self.parse_packet_sent_chunked(config.path)
            losses = self.parse_packet_lost_chunked(config.path)
            
            all_metrics_raw.append(metrics)
            all_packets_raw.append(packets)
            all_losses_raw.append(losses)
            
            # Force garbage collection after each large file
            gc.collect()
        
        # Align times
        print("\nAligning timelines...")
        all_combined_raw = all_metrics_raw + all_packets_raw + all_losses_raw
        all_combined_aligned = self.align_times(all_combined_raw)
        
        # Split back into separate lists
        n_files = len(configs)
        all_metrics_raw = all_combined_aligned[:n_files]
        all_packets_raw = all_combined_aligned[n_files:2*n_files]
        all_losses_raw = all_combined_aligned[2*n_files:3*n_files]
        
        # Separate paths and fill missing values
        print("Separating paths and processing data...")
        for i, config in enumerate(configs):
            metrics_by_path = self.separate_paths(all_metrics_raw[i])
            packets_by_path = self.separate_paths(all_packets_raw[i])
            losses_by_path = self.separate_paths(all_losses_raw[i])
            
            # Fill missing values for each path
            for path_id in self.paths_to_plot:
                if path_id in metrics_by_path:
                    metrics_by_path[path_id] = self.fill_missing_values(metrics_by_path[path_id])
            
            all_metrics.append(metrics_by_path)
            all_packets.append(packets_by_path)
            all_losses.append(losses_by_path)
            
            print(f"{config.label} - Path distribution:")
            for path_id in self.paths_to_plot:
                m_count = len(metrics_by_path.get(path_id, []))
                p_count = len(packets_by_path.get(path_id, []))
                l_count = len(losses_by_path.get(path_id, []))
                print(f"  Path{path_id}: {m_count} metrics, {p_count} packets, {l_count} losses")
        
        # Print packet loss summary
        self.print_packet_loss_summary(all_losses, configs)
        
        # Compute utilization metrics
        print(f"=== CWND Utilization Analysis ===")
        for i, config in enumerate(configs):
            for path_id in self.paths_to_plot:
                metrics_data = all_metrics[i].get(path_id, pd.DataFrame())
                packet_data = all_packets[i].get(path_id, pd.DataFrame())
                self.compute_utilization_metrics(metrics_data, packet_data, 
                                               config.label, path_id)
        
        # Generate plots with packet loss markers
        print("Generating plots with packet loss indicators...")
        self.plot_congestion_window(all_metrics, all_losses, configs, output_prefix)
        self.plot_bytes_in_flight(all_metrics, all_losses, configs, output_prefix)
        self.plot_rtt_comparison(all_metrics, all_losses, configs, output_prefix)
        self.plot_send_rate(all_packets, all_losses, configs, output_prefix)
        plt.show()
        print("Analysis complete!")


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