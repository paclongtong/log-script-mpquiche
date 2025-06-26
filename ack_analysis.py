#!/usr/bin/env python3
"""
QUIC Multipath ACK Analysis Script
Analyzes ACK and MP_ACK frame patterns in server qlog files from multipath QUIC experiments.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path

def parse_ack_frames_from_packet_sent(path):
    """
    Parse transport:packet_sent events and extract ACK and MP_ACK frame information.
    
    Args:
        path: Path to the qlog file
        
    Returns:
        DataFrame with ACK frame data including timestamps, sizes, and path information
    """
    records = []
    
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line.startswith('{'):
                continue
                
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {line_num}: {e}")
                continue
                
            # Focus on packet_sent events
            if event.get('name') != 'transport:packet_sent':
                continue
                
            timestamp = event.get('time')
            if timestamp is None:
                continue
                
            data = event.get('data', {})
            path_id = data.get('path_id', 0)
            
            # Extract frames from the packet
            frames = data.get('frames', [])
            if not frames:
                continue
                
            # Process each frame in the packet
            for frame in frames:
                frame_type = frame.get('frame_type', '')
                
                # Check for ACK or MP_ACK frames
                if frame_type in ['ack', 'mp_ack']:
                    # Calculate frame size (approximation based on ACK ranges)
                    frame_size = estimate_ack_frame_size(frame)
                    
                    record = {
                        'time': timestamp,
                        'frame_type': frame_type,
                        'path_id': path_id,
                        'frame_size': frame_size,
                        'ack_delay': frame.get('ack_delay', 0),
                        'acked_ranges': len(frame.get('acked_ranges', [])),
                        'largest_acked': frame.get('largest_acked', 0)
                    }
                    
                    # Add MP_ACK specific fields
                    if frame_type == 'mp_ack':
                        record['destination_connection_id'] = frame.get('destination_connection_id', '')
                        record['mp_ack_ranges'] = len(frame.get('mp_ack_ranges', []))
                    
                    records.append(record)
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('time').reset_index(drop=True)
        # Convert time to relative seconds from start
        df['relative_time'] = (df['time'] - df['time'].min()) / 1000  # Convert to seconds
    
    return df

def estimate_ack_frame_size(frame):
    """
    Estimate the size of an ACK frame based on its contents.
    This is an approximation based on QUIC ACK frame format.
    """
    base_size = 3  # Frame type + largest_acked + ack_delay base encoding
    
    # Add size for acked ranges
    acked_ranges = frame.get('acked_ranges', [])
    ranges_size = len(acked_ranges) * 4  # Approximate encoding per range
    
    # Add size for ACK delay
    ack_delay = frame.get('ack_delay', 0)
    # Convert to int if it's a float, handle the bit_length calculation properly
    try:
        if isinstance(ack_delay, (int, float)):
            ack_delay_int = int(ack_delay) if ack_delay >= 0 else 0
            delay_size = max(1, (ack_delay_int.bit_length() + 7) // 8) if ack_delay_int > 0 else 1
        else:
            delay_size = 1  # Default size if ack_delay is not a number
    except (AttributeError, ValueError):
        delay_size = 1  # Fallback to default size
    
    # MP_ACK has additional overhead
    if frame.get('frame_type') == 'mp_ack':
        base_size += 8  # Additional connection ID field
        mp_ranges = frame.get('mp_ack_ranges', [])
        ranges_size += len(mp_ranges) * 4
    
    return base_size + ranges_size + delay_size

def analyze_ack_patterns(df):
    """
    Perform comprehensive analysis of ACK patterns.
    """
    if df.empty:
        print("No ACK data found for analysis")
        return {}
    
    analysis = {}
    
    # Overall statistics
    analysis['total_ack_frames'] = len(df)
    analysis['total_ack_bytes'] = df['frame_size'].sum()
    analysis['avg_ack_size'] = df['frame_size'].mean()
    analysis['median_ack_size'] = df['frame_size'].median()
    
    # Frame type distribution
    frame_type_stats = df.groupby('frame_type').agg({
        'frame_size': ['count', 'sum', 'mean', 'std'],
        'ack_delay': 'mean'
    }).round(2)
    analysis['frame_type_stats'] = frame_type_stats
    
    # Path-wise analysis
    if 'path_id' in df.columns:
        path_stats = df.groupby(['path_id', 'frame_type']).agg({
            'frame_size': ['count', 'sum', 'mean'],
            'relative_time': ['min', 'max']
        }).round(2)
        analysis['path_stats'] = path_stats
    
    # Time-based analysis
    df['time_bucket'] = pd.cut(df['relative_time'], bins=20, labels=False)
    time_analysis = df.groupby('time_bucket').agg({
        'frame_size': ['count', 'sum'],
        'relative_time': 'mean'
    }).round(2)
    analysis['time_distribution'] = time_analysis
    
    # ACK efficiency metrics
    if len(df) > 0:
        duration = df['relative_time'].max() - df['relative_time'].min()
        analysis['ack_rate_per_second'] = len(df) / duration if duration > 0 else 0
        analysis['ack_bytes_per_second'] = df['frame_size'].sum() / duration if duration > 0 else 0
    
    return analysis

def create_visualizations(df, output_dir="./ack_analysis_plots"):
    """
    Create comprehensive visualizations of ACK patterns.
    """
    if df.empty:
        print("No data to visualize")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. ACK Frame Size Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QUIC Multipath ACK Analysis - Server Perspective', fontsize=16, fontweight='bold')
    
    # Frame size histogram
    axes[0, 0].hist(df['frame_size'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('ACK Frame Size Distribution')
    axes[0, 0].set_xlabel('Frame Size (bytes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frame type comparison
    if 'frame_type' in df.columns:
        frame_sizes = [df[df['frame_type'] == ft]['frame_size'].values for ft in df['frame_type'].unique()]
        axes[0, 1].boxplot(frame_sizes, tick_labels=df['frame_type'].unique())
        axes[0, 1].set_title('Frame Size by Type')
        axes[0, 1].set_ylabel('Frame Size (bytes)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Time series of ACK traffic
    time_grouped = df.groupby(pd.cut(df['relative_time'], bins=50), observed=False)['frame_size'].sum()
    time_points = [interval.mid for interval in time_grouped.index]
    axes[1, 0].plot(time_points, time_grouped.values, marker='o', markersize=3)
    axes[1, 0].set_title('ACK Traffic Over Time')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Total ACK Bytes per Interval')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Path distribution (if multipath)
    if 'path_id' in df.columns and df['path_id'].nunique() > 1:
        path_data = df.groupby('path_id')['frame_size'].sum()
        axes[1, 1].pie(path_data.values, labels=[f'Path {pid}' for pid in path_data.index], 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('ACK Traffic Distribution by Path')
    else:
        # ACK delay distribution
        if 'ack_delay' in df.columns:
            axes[1, 1].hist(df['ack_delay'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('ACK Delay Distribution')
            axes[1, 1].set_xlabel('ACK Delay (microseconds)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ack_analysis_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed time series analysis
    if len(df) > 100:  # Only if we have enough data points
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('ACK Traffic Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Rolling window analysis
        window_size = max(10, len(df) // 100)
        df_sorted = df.sort_values('relative_time')
        
        # ACK rate over time - use numeric window instead of time-based
        rolling_count = df_sorted['frame_size'].rolling(window=window_size, min_periods=1).count()
        axes[0].plot(df_sorted['relative_time'], rolling_count.values)
        axes[0].set_title(f'ACK Frame Rate (Rolling {window_size} frames window)')
        axes[0].set_ylabel('ACKs per window')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative ACK bytes
        axes[1].plot(df_sorted['relative_time'], df_sorted['frame_size'].cumsum())
        axes[1].set_title('Cumulative ACK Bytes Over Time')
        axes[1].set_ylabel('Cumulative Bytes')
        axes[1].grid(True, alpha=0.3)
        
        # ACK size variation over time
        rolling_size = df_sorted['frame_size'].rolling(window=window_size, min_periods=1).mean()
        axes[2].plot(df_sorted['relative_time'], rolling_size.values)
        axes[2].set_title(f'Average ACK Size (Rolling {window_size} frames window)')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Average Frame Size (bytes)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ack_time_series.png", dpi=300, bbox_inches='tight')
        plt.show()

def print_analysis_summary(analysis):
    """
    Print a comprehensive summary of the ACK analysis.
    """
    print("=" * 60)
    print("QUIC MULTIPATH ACK ANALYSIS SUMMARY (Server Perspective)")
    print("=" * 60)
    
    print(f"Total ACK Frames: {analysis.get('total_ack_frames', 0):,}")
    print(f"Total ACK Traffic: {analysis.get('total_ack_bytes', 0):,.0f} bytes")
    print(f"Average ACK Size: {analysis.get('avg_ack_size', 0):.2f} bytes")
    print(f"Median ACK Size: {analysis.get('median_ack_size', 0):.2f} bytes")
    
    if 'ack_rate_per_second' in analysis:
        print(f"ACK Rate: {analysis['ack_rate_per_second']:.2f} frames/second")
        print(f"ACK Bandwidth: {analysis['ack_bytes_per_second']:.2f} bytes/second")
    
    print("\n" + "-" * 40)
    print("FRAME TYPE BREAKDOWN:")
    print("-" * 40)
    
    if 'frame_type_stats' in analysis and not analysis['frame_type_stats'].empty:
        print(analysis['frame_type_stats'])
    
    if 'path_stats' in analysis and not analysis['path_stats'].empty:
        print("\n" + "-" * 40)
        print("PATH-WISE STATISTICS:")
        print("-" * 40)
        print(analysis['path_stats'])

def main():
    parser = argparse.ArgumentParser(description='Analyze ACK patterns in QUIC multipath qlog files')
    parser.add_argument('qlog_file', help='Path to the server qlog file')
    parser.add_argument('--output-dir', default='./plots', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    print(f"Parsing ACK data from: {args.qlog_file}")
    
    # Parse the qlog file
    df = parse_ack_frames_from_packet_sent(args.qlog_file)
    
    if df.empty:
        print("No ACK frames found in the qlog file!")
        return
    
    print(f"Found {len(df)} ACK frames")
    
    # Analyze patterns
    analysis = analyze_ack_patterns(df)
    
    # Print summary
    print_analysis_summary(analysis)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(df, args.output_dir)
        print(f"\nPlots saved to: {args.output_dir}")
    
    # Save detailed data to CSV
    output_file = Path(args.output_dir) / "ack_data_detailed.csv"
    df.to_csv(output_file, index=False)
    print(f"Detailed data saved to: {output_file}")

if __name__ == "__main__":
    main()