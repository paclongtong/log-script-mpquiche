import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

def parse_min_rtt_updates(path: str) -> pd.DataFrame:
    """
    Parse a QLOG file and extract min_rtt updates over time.
    Only records entries where min_rtt actually changes.
    """
    records = []
    last_min_rtt_by_path = {}  # Track last seen min_rtt per path_id
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip non-JSON lines (handles manual text additions)
            if not line.startswith('{'):
                continue
                
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON objects
                continue
                
            # Only process recovery:metrics_updated events
            if event.get("name") != "recovery:metrics_updated":
                continue
                
            data = event.get("data", {})
            time = event.get("time")
            path_id = data.get("path_id")
            min_rtt = data.get("min_rtt")
            
            # Skip if missing mandatory fields
            if time is None or path_id is None or min_rtt is None:
                continue
            
            # Check if min_rtt has changed for this path
            if path_id not in last_min_rtt_by_path or last_min_rtt_by_path[path_id] != min_rtt:
                # Min RTT updated! Record this event
                record = {
                    "time_s": time,
                    "path_id": path_id,
                    "min_rtt": min_rtt,
                    "smoothed_rtt": data.get("smoothed_rtt"),
                    "latest_rtt": data.get("latest_rtt"),
                    "rtt_variance": data.get("rtt_variance"),
                    "congestion_window": data.get("congestion_window"),
                    "bytes_in_flight": data.get("bytes_in_flight"),
                    "pacing_rate": data.get("pacing_rate")
                }
                
                # Calculate change from previous min_rtt if available
                if path_id in last_min_rtt_by_path:
                    record["min_rtt_change"] = min_rtt - last_min_rtt_by_path[path_id]
                    record["min_rtt_change_pct"] = ((min_rtt - last_min_rtt_by_path[path_id]) / 
                                                   last_min_rtt_by_path[path_id] * 100)
                else:
                    record["min_rtt_change"] = None
                    record["min_rtt_change_pct"] = None
                
                records.append(record)
                last_min_rtt_by_path[path_id] = min_rtt
    
    if not records:
        print("No min_rtt updates found in the qlog file")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    return df

def analyze_min_rtt_trends(df: pd.DataFrame) -> None:
    """
    Analyze and print min_rtt trends from the parsed data.
    """
    if df.empty:
        print("No data to analyze")
        return
    
    print("=== Min RTT Update Analysis ===")
    print(f"Total min_rtt updates: {len(df)}")
    print(f"Time range: {df['time_s'].min():.3f}s to {df['time_s'].max():.3f}s")
    print(f"Duration: {df['time_s'].max() - df['time_s'].min():.3f}s")
    
    # Per-path analysis
    for path_id in sorted(df['path_id'].unique()):
        path_df = df[df['path_id'] == path_id]
        print(f"\n--- Path {path_id} ---")
        print(f"Updates: {len(path_df)}")
        print(f"Min RTT range: {path_df['min_rtt'].min():.3f}ms to {path_df['min_rtt'].max():.3f}ms")
        
        # Show first few and last few updates
        print("\nFirst 3 updates:")
        for _, row in path_df.head(3).iterrows():
            change_str = ""
            if row['min_rtt_change'] is not None:
                change_str = f" (Δ{row['min_rtt_change']:+.3f}ms, {row['min_rtt_change_pct']:+.1f}%)"
            print(f"  t={row['time_s']:8.3f}s: {row['min_rtt']:7.3f}ms{change_str}")
        
        if len(path_df) > 6:
            print("  ...")
            print("Last 3 updates:")
            for _, row in path_df.tail(3).iterrows():
                change_str = f" (Δ{row['min_rtt_change']:+.3f}ms, {row['min_rtt_change_pct']:+.1f}%)"
                print(f"  t={row['time_s']:8.3f}s: {row['min_rtt']:7.3f}ms{change_str}")

def plot_min_rtt_trends(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot min_rtt trends over time.
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Min RTT over time
    ax1 = axes[0]
    for path_id in sorted(df['path_id'].unique()):
        path_df = df[df['path_id'] == path_id]
        ax1.plot(path_df['time_s'], path_df['min_rtt'], 
                marker='o', linestyle='-', label=f'Path {path_id}', markersize=4)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Min RTT (ms)')
    ax1.set_title('Min RTT Updates Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Min RTT changes (deltas)
    ax2 = axes[1]
    for path_id in sorted(df['path_id'].unique()):
        path_df = df[df['path_id'] == path_id]
        # Skip first point (no change data)
        change_df = path_df[path_df['min_rtt_change'].notna()]
        if not change_df.empty:
            ax2.scatter(change_df['time_s'], change_df['min_rtt_change'], 
                       label=f'Path {path_id}', alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Min RTT Change (ms)')
    ax2.set_title('Min RTT Changes Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    """
    Main function to run the min_rtt analysis.
    """
    # Example usage
    qlog_path = "/tmp/minitopo_experiences/server-75594b763df0ec97d899fcb327a93cac869c1187.sqlog"
    
    try:
        print(f"Parsing qlog file: {qlog_path}")
        df = parse_min_rtt_updates(qlog_path)
        
        if df.empty:
            return
        
        # Print analysis
        analyze_min_rtt_trends(df)
        

        plot_min_rtt_trends(df)
        
        # # Ask if user wants to save CSV
        # csv_choice = input("\nSave data to CSV? Enter filename (or press Enter to skip): ").strip()
        # if csv_choice:
        #     df.to_csv(csv_choice, index=False)
        #     print(f"Data saved to: {csv_choice}")
            
    except FileNotFoundError:
        print(f"Error: File '{qlog_path}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()