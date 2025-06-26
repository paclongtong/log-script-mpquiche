import json
import pandas as pd
import matplotlib.pyplot as plt

# === Configurable switches ===
PLOT_HEADROOM = False                 # Plot headroom ratio (cwnd vs. bytes_in_flight)
PLOT_LATEST_RTT = True               # Plot latest RTT (overall)
PLOT_LATEST_RTT_PER_PATH = True     # Split latest RTT by path_id
PLOT_RTT_VARIANCE = False            # Plot RTT variance
PLOT_BYTES_IN_FLIGHT_PER_PATH = True   
PLOT_BYTES_IN_FLIGHT = True
PLOT_CWND = True
PLOT_PAYLOAD_LENGTH = False
CALC_PAYLOAD_STATS = False
PLOT_PACING_RATE = True

# Default files and labels
QLOG_FILE_1 = "/tmp/minitopo_experiences/client-6195d17444700050b091f035b66c09cca3e8ba2b.sqlog"
LABEL_1 = "client"
QLOG_FILE_2 = "/tmp/minitopo_experiences/server-25cb5c459201cbd2803c231011a95da7709ed1ff.sqlog"
LABEL_2 = "server"

user_defined_label = "original"

    
def parse_qlog(path):
    """
    Parse a QLOG file and extract time, cwnd, bytes_in_flight, latest_rtt, rtt_variance, and path_id.
    """
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # skip lines that are not valid JSON objects
                continue
            if event.get("name") == "recovery:metrics_updated":
                data = event.get("data", {})
                time = event.get("time")
                path_id = data.get("path_id")
                latest_rtt = data.get("latest_rtt")
                rttvar = data.get("rtt_variance")
                bif = data.get("bytes_in_flight")
                cwnd = data.get("congestion_window")
                pacing = data.get("pacing_rate") 
                
                # Only require time, path_id, latest_rtt, and bytes_in_flight as mandatory
                # cwnd and other fields are optional
                if time is None or path_id is None or latest_rtt is None or bif is None:
                    # skip records missing mandatory fields
                    continue
                    
                record = {
                    "time_s": time,
                    "path_id": path_id,
                    "latest_rtt": latest_rtt,
                    "rtt_variance": rttvar,
                    "bytes_in_flight": bif
                }
                # only include cwnd if present
                if cwnd is not None:    
                    record["cwnd"] = cwnd
                if pacing is not None:                       # <<< only if present
                    record["pacing_rate"] = pacing
                records.append(record)
    df = pd.DataFrame(records)
    # compute headroom ratio when possible (only for records that have cwnd)
    if "cwnd" in df.columns and not df["cwnd"].isna().all():
        # Only compute headroom for rows where cwnd is not null
        df["headroom_ratio"] = ((df["cwnd"] - df["bytes_in_flight"]) / df["cwnd"]).where(df["cwnd"].notna())
    return df

def parse_payload_qlog(path):
    """
    Parse transport:packet_sent events for payload_length into a DataFrame.
    """
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # skip lines that are not valid JSON objects
                continue
            if event.get("name") == "transport:packet_sent":
                data = event.get("data", {})
                raw = data.get("raw", {})
                pl = raw.get("payload_length")
                time_s = event.get("time")
                if time_s is None or pl is None:
                    continue
                records.append({"time_s": time_s, "payload_length": pl})
    return pd.DataFrame(records)

def compute_payload_stats(df, label):
    """
    Compute distribution statistics for payload_length DataFrame.
    Prints count, mean, median, mode, std, and percentiles.
    """
    if df.empty:
        print(f"{label}: No payload_length data found.")
        return
    stats = df["payload_length"].describe()
    mode_vals = df["payload_length"].mode().tolist()
    print(f"--- Payload Length Stats for {label} {user_defined_label}---")
    print(f"Count: {int(stats['count'])}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['50%']:.2f}")
    print(f"Mode: {mode_vals}")
    print(f"Std: {stats['std']:.2f}")
    print(f"Min: {stats['min']:.2f}, 25%: {stats['25%']:.2f}, 75%: {stats['75%']:.2f}, Max: {stats['max']:.2f}\n")

def plot_trends(qlog1, label1, qlog2, label2):
    df1 = parse_qlog(qlog1)
    df2 = parse_qlog(qlog2)

    print(f"Data range for {label1}: {df1['time_s'].min():.2f}s to {df1['time_s'].max():.2f}s ({len(df1)} records)")
    print(f"Data range for {label2}: {df2['time_s'].min():.2f}s to {df2['time_s'].max():.2f}s ({len(df2)} records)")

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15)) # Adjust figsize as needed
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
    plot_idx = 0

    # 1) Headroom Ratio
    if PLOT_HEADROOM and "headroom_ratio" in df1.columns and "headroom_ratio" in df2.columns:
        ax = axes[plot_idx]
        df1_valid = df1.dropna(subset=['headroom_ratio'])
        df2_valid = df2.dropna(subset=['headroom_ratio'])
        if not df1_valid.empty:
            ax.plot(df1_valid["time_s"], df1_valid["headroom_ratio"], label=label1)
        if not df2_valid.empty:
            ax.plot(df2_valid["time_s"], df2_valid["headroom_ratio"], label=label2)
        ax.set_title("Headroom Ratio Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Headroom Ratio")
        ax.legend()
        plot_idx += 1

    # 2) Latest RTT
    if PLOT_LATEST_RTT:
        ax = axes[plot_idx]
        if PLOT_LATEST_RTT_PER_PATH:
            for pid in sorted(df1["path_id"].unique()):
                d = df1[df1["path_id"] == pid]
                ax.plot(d["time_s"], d["latest_rtt"], label=f"{label1}-path{pid}")
            for pid in sorted(df2["path_id"].unique()):
                d = df2[df2["path_id"] == pid]
                ax.plot(d["time_s"], d["latest_rtt"], label=f"{label2}-path{pid}")
        else:
            ax.plot(df1["time_s"], df1["latest_rtt"], label=label1)
            ax.plot(df2["time_s"], df2["latest_rtt"], label=label2)
        ax.set_title("Latest RTT Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Latest RTT (ms)")
        ax.legend()
        plot_idx += 1

    # 3) RTT Variance
    if PLOT_RTT_VARIANCE:
        ax = axes[plot_idx]
        df1_rttvar = df1.dropna(subset=['rtt_variance'])
        df2_rttvar = df2.dropna(subset=['rtt_variance'])
        if not df1_rttvar.empty:
            ax.plot(df1_rttvar["time_s"], df1_rttvar["rtt_variance"], label=label1)
        if not df2_rttvar.empty:
            ax.plot(df2_rttvar["time_s"], df2_rttvar["rtt_variance"], label=label2)
        ax.set_title("RTT Variance Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RTT Variance (ms^2)")
        ax.legend()
        plot_idx += 1

    # 4) Bytes In Flight
    if PLOT_BYTES_IN_FLIGHT:
        ax = axes[plot_idx]
        if PLOT_BYTES_IN_FLIGHT_PER_PATH:
            for pid in sorted(df1["path_id"].unique()):
                d = df1[df1["path_id"] == pid]
                ax.plot(d["time_s"], d["bytes_in_flight"], label=f"{label1}-path{pid}")
            for pid in sorted(df2["path_id"].unique()):
                d = df2[df2["path_id"] == pid]
                ax.plot(d["time_s"], d["bytes_in_flight"], label=f"{label2}-path{pid}")
        else:
            ax.plot(df1["time_s"], df1["bytes_in_flight"], label=label1)
            ax.plot(df2["time_s"], df2["bytes_in_flight"], label=label2)
        ax.set_title("Bytes In Flight Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bytes In Flight")
        ax.legend()
        plot_idx += 1

    # 5) Congestion Window
    if PLOT_CWND:
        ax = axes[plot_idx]
        cwnd_plotted = False
        for pid in sorted(df1["path_id"].unique()):
            d = df1[df1["path_id"] == pid]
            d_cwnd = d.dropna(subset=['cwnd'])
            if not d_cwnd.empty:
                ax.plot(d_cwnd["time_s"], d_cwnd["cwnd"], label=f"{label1}-path{pid}")
                cwnd_plotted = True
        for pid in sorted(df2["path_id"].unique()):
            d = df2[df2["path_id"] == pid]
            d_cwnd = d.dropna(subset=['cwnd'])
            if not d_cwnd.empty:
                ax.plot(d_cwnd["time_s"], d_cwnd["cwnd"], label=f"{label2}-path{pid}")
                cwnd_plotted = True
        
        if cwnd_plotted:
            ax.set_title("Congestion Window Over Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Congestion Window (bytes)")
            ax.legend()
        else:
            # Clear the current subplot if no data to plot
            fig.delaxes(ax) 
            print("Warning: No congestion window data found; skipping CWND plot in combined view.")
        plot_idx += 1
    
    # 6) Payload Length
    if PLOT_PAYLOAD_LENGTH:
        ax = axes[plot_idx]
        dfp1 = parse_payload_qlog(qlog1)
        dfp2 = parse_payload_qlog(qlog2)
        if not dfp1.empty or not dfp2.empty:
            if not dfp1.empty:
                ax.plot(dfp1["time_s"], dfp1["payload_length"], label=label1)
            if not dfp2.empty:
                ax.plot(dfp2["time_s"], dfp2["payload_length"], label=label2)
            ax.set_title("Payload Length Over Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Payload Length (bytes)")
            ax.legend()
        else:
            fig.delaxes(ax)
            print("Warning: No payload_length data found; skipping payload plot in combined view.")
        plot_idx += 1

    # 7) Pacing Rate
    if PLOT_PACING_RATE:
        ax = axes[plot_idx]
        pacing_plotted = False
        if 'pacing_rate' in df1.columns:
            df1_pacing = df1.dropna(subset=['pacing_rate'])
            if not df1_pacing.empty:
                for pid in sorted(df1_pacing["path_id"].unique()):
                    d = df1_pacing[df1_pacing["path_id"] == pid]
                    ax.plot(d["time_s"], d["pacing_rate"], label=f"{label1}-path{pid}")
                    pacing_plotted = True
        
        if 'pacing_rate' in df2.columns:
            df2_pacing = df2.dropna(subset=['pacing_rate'])
            if not df2_pacing.empty:
                for pid in sorted(df2_pacing["path_id"].unique()):
                    d = df2_pacing[df2_pacing["path_id"] == pid]
                    ax.plot(d["time_s"], d["pacing_rate"], label=f"{label2}-path{pid}")
                    pacing_plotted = True
        
        if pacing_plotted:
            ax.set_title("Pacing Rate Over Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pacing Rate (bytes/s)")
            ax.legend()
        else:
            fig.delaxes(ax)
            print("Warning: No pacing rate data found; skipping pacing rate plot in combined view.")
        plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    # plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.suptitle(f"Combined QLOG Metrics for {user_defined_label}", y=1.02, fontsize=16) # Add a super title
    plt.show()

# The existing individual plot calls should remain if you still want them
# separate from the combined plot, or you can remove them if the combined
# plot is sufficient.
def plot_individual_trends(qlog1, label1, qlog2, label2):
    df1 = parse_qlog(qlog1)
    df2 = parse_qlog(qlog2)

    # 1) Headroom Ratio
    if PLOT_HEADROOM and "headroom_ratio" in df1.columns and "headroom_ratio" in df2.columns:
        plt.figure()
        # Only plot non-null headroom ratios
        df1_valid = df1.dropna(subset=['headroom_ratio'])
        df2_valid = df2.dropna(subset=['headroom_ratio'])
        if not df1_valid.empty:
            plt.plot(df1_valid["time_s"], df1_valid["headroom_ratio"], label=label1)
        if not df2_valid.empty:
            plt.plot(df2_valid["time_s"], df2_valid["headroom_ratio"], label=label2)
        plt.title("Headroom Ratio Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Headroom Ratio")
        plt.legend()

    # 2) Latest RTT
    if PLOT_LATEST_RTT:
        plt.figure()
        if PLOT_LATEST_RTT_PER_PATH:
            # plot each path separately
            for pid in sorted(df1["path_id"].unique()):
                d = df1[df1["path_id"] == pid]
                plt.plot(d["time_s"], d["latest_rtt"], label=f"{label1}-path{pid}")
            for pid in sorted(df2["path_id"].unique()):
                d = df2[df2["path_id"] == pid]
                plt.plot(d["time_s"], d["latest_rtt"], label=f"{label2}-path{pid}")
        else:
            # overall latest RTT (ignores path_id)
            plt.plot(df1["time_s"], df1["latest_rtt"], label=label1)
            plt.plot(df2["time_s"], df2["latest_rtt"], label=label2)
        plt.title("Latest RTT Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Latest RTT (ms)")
        plt.legend()

    # 3) RTT Variance
    if PLOT_RTT_VARIANCE:
        plt.figure()
        # Filter out null values for RTT variance
        df1_rttvar = df1.dropna(subset=['rtt_variance'])
        df2_rttvar = df2.dropna(subset=['rtt_variance'])
        if not df1_rttvar.empty:
            plt.plot(df1_rttvar["time_s"], df1_rttvar["rtt_variance"], label=label1)
        if not df2_rttvar.empty:
            plt.plot(df2_rttvar["time_s"], df2_rttvar["rtt_variance"], label=label2)
        plt.title("RTT Variance Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("RTT Variance (ms^2)")
        plt.legend()

    # 4) Bytes In Flight
    if PLOT_BYTES_IN_FLIGHT:
        plt.figure()
        if PLOT_BYTES_IN_FLIGHT_PER_PATH:
            for pid in sorted(df1["path_id"].unique()):
                d = df1[df1["path_id"] == pid]
                plt.plot(d["time_s"], d["bytes_in_flight"], label=f"{label1}-path{pid}")
            for pid in sorted(df2["path_id"].unique()):
                d = df2[df2["path_id"] == pid]
                plt.plot(d["time_s"], d["bytes_in_flight"], label=f"{label2}-path{pid}")
        else:
            plt.plot(df1["time_s"], df1["bytes_in_flight"], label=label1)
            plt.plot(df2["time_s"], df2["bytes_in_flight"], label=label2)
        plt.title("Bytes In Flight Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Bytes In Flight")
        plt.legend()

    # 5) Congestion Window
    if PLOT_CWND:
        plt.figure()
        # Only plot paths that have cwnd data
        cwnd_plotted = False
        for pid in sorted(df1["path_id"].unique()):
            d = df1[df1["path_id"] == pid]
            d_cwnd = d.dropna(subset=['cwnd'])
            if not d_cwnd.empty:
                plt.plot(d_cwnd["time_s"], d_cwnd["cwnd"], label=f"{label1}-path{pid}")
                cwnd_plotted = True
        for pid in sorted(df2["path_id"].unique()):
            d = df2[df2["path_id"] == pid]
            d_cwnd = d.dropna(subset=['cwnd'])
            if not d_cwnd.empty:
                plt.plot(d_cwnd["time_s"], d_cwnd["cwnd"], label=f"{label2}-path{pid}")
                cwnd_plotted = True
        
        if cwnd_plotted:
            plt.title("Congestion Window Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Congestion Window (bytes)")
            plt.legend()
        else:
            plt.close()  # Close the empty figure
            print("Warning: No congestion window data found; skipping CWND plot.")
    
    # 6) Payload Length
    if PLOT_PAYLOAD_LENGTH:
        dfp1 = parse_payload_qlog(qlog1)
        dfp2 = parse_payload_qlog(qlog2)
        if not dfp1.empty or not dfp2.empty:
            plt.figure()
            if not dfp1.empty:
                plt.plot(dfp1["time_s"], dfp1["payload_length"], label=label1)
            if not dfp2.empty:
                plt.plot(dfp2["time_s"], dfp2["payload_length"], label=label2)
            plt.title("Payload Length Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Payload Length (bytes)")
            plt.legend()
        else:
            print("Warning: No payload_length data found; skipping payload plot.")

    # 7) Pacing Rate
    if PLOT_PACING_RATE:
        plt.figure()
        pacing_plotted = False
        # Check if df1 has pacing rate data
        if 'pacing_rate' in df1.columns:
            df1_pacing = df1.dropna(subset=['pacing_rate'])
            if not df1_pacing.empty:
                for pid in sorted(df1_pacing["path_id"].unique()):
                    d = df1_pacing[df1_pacing["path_id"] == pid]
                    plt.plot(d["time_s"], d["pacing_rate"], label=f"{label1}-path{pid}")
                    pacing_plotted = True
        
        # Check if df2 has pacing rate data
        if 'pacing_rate' in df2.columns:
            df2_pacing = df2.dropna(subset=['pacing_rate'])
            if not df2_pacing.empty:
                for pid in sorted(df2_pacing["path_id"].unique()):
                    d = df2_pacing[df2_pacing["path_id"] == pid]
                    plt.plot(d["time_s"], d["pacing_rate"], label=f"{label2}-path{pid}")
                    pacing_plotted = True
        
        if pacing_plotted:
            plt.title("Pacing Rate Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Pacing Rate (bytes/s)")
            plt.legend()
        else:
            plt.close()  # Close the empty figure
            print("Warning: No pacing rate data found; skipping pacing rate plot.")

    plt.show()


if __name__ == "__main__":
    # 1) Compute payload distribution stats 
    if CALC_PAYLOAD_STATS:
        dfp1 = parse_payload_qlog(QLOG_FILE_1)
        dfp2 = parse_payload_qlog(QLOG_FILE_2)
        compute_payload_stats(dfp1, LABEL_1)
        compute_payload_stats(dfp2, LABEL_2)

    # Call the new combined plot function
    plot_trends(QLOG_FILE_1, LABEL_1, QLOG_FILE_2, LABEL_2)

    # Call the original individual plot function (optional, if you still want individual plots)
    # plot_individual_trends(QLOG_FILE_1, LABEL_1, QLOG_FILE_2, LABEL_2)