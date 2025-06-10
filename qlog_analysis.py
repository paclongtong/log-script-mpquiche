# /home/paul/data_quiche/qlog/ori/server-b8191a9c9180012653dd6fb1a00a5c7c3f90cd54.sqlog
# /home/paul/data_quiche/qlog/dev/server-348198ca5e4e900a6732e14cc9cea57b2bee5f76.sqlog

'''
import json
import pandas as pd
import matplotlib.pyplot as plt

# === Hardcoded default files and labels ===
# QLOG_FILE_1 = "/home/paul/data_quiche/qlog/ori/server-7955d219ce9edaae849637938611addc7984ebef.sqlog"
QLOG_FILE_1 = "/home/paul/data_quiche/qlog/ori/client-af6c969abb712c0244939e8d3075363ecc9418bb.sqlog"

LABEL_1 = "original"
# QLOG_FILE_2 = "/home/paul/data_quiche/qlog/dev/server-310b25b6a1d5522b643dfcd7c864d7e377ca5f3d.sqlog"
QLOG_FILE_2 = "/home/paul/data_quiche/qlog/dev/client-b6c980769c4b7147c4db43b85efbeac6212029a0.sqlog"
LABEL_2 = "separate"

def parse_qlog(path):
    """Parse a QLOG file and extract time, cwnd, bytes_in_flight, and rtt_variance."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            event = json.loads(line)
            if event.get("name") == "recovery:metrics_updated":
                data = event.get("data", {})
                time = event.get("time")
                cwnd = data.get("congestion_window")
                bif = data.get("bytes_in_flight")
                rttvar = data.get("rtt_variance")
                latest_rtt = data.get("latest_rtt")
                if time is not None and cwnd is not None and bif is not None and latest_rtt is not None:
                    records.append({
                        "time_s": time,
                        "cwnd": cwnd,
                        "bytes_in_flight": bif,
                        "latest_rtt": latest_rtt
                    })
    df = pd.DataFrame(records)
    df["headroom_ratio"] = (df["cwnd"] - df["bytes_in_flight"]) / df["cwnd"]
    return df

def plot_trends(qlog1, label1, qlog2, label2):
    df1 = parse_qlog(qlog1)
    df2 = parse_qlog(qlog2)

    # Plot headroom_ratio
    plt.figure()
    plt.plot(df1["time_s"], df1["headroom_ratio"], label=label1)
    plt.plot(df2["time_s"], df2["headroom_ratio"], label=label2)
    plt.title("Headroom Ratio Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Headroom Ratio")
    plt.legend()

    # Plot RTT variance
    plt.figure()
    plt.plot(df1["time_s"], df1["latest_rtt"], label=label1)
    plt.plot(df2["time_s"], df2["latest_rtt"], label=label2)
    plt.title("Latest RTT Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Latest RTT (ms)")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    plot_trends(QLOG_FILE_1, LABEL_1, QLOG_FILE_2, LABEL_2)
'''


import json
import pandas as pd
import matplotlib.pyplot as plt

# === Configurable switches ===
PLOT_HEADROOM = False                 # Plot headroom ratio (cwnd vs. bytes_in_flight)
PLOT_LATEST_RTT = True               # Plot latest RTT (overall)
PLOT_LATEST_RTT_PER_PATH = True     # Split latest RTT by path_id
PLOT_RTT_VARIANCE = True            # Plot RTT variance
PLOT_BYTES_IN_FLIGHT_PER_PATH = True   
PLOT_BYTES_IN_FLIGHT = True
PLOT_CWND = True
PLOT_PAYLOAD_LENGTH = False
CALC_PAYLOAD_STATS = False
PLOT_PACING_RATE = True

# Default files and labels
# QLOG_FILE_1 = "/home/paul/data_quiche/qlog/ori/client-af6c969abb712c0244939e8d3075363ecc9418bb.sqlog"
# QLOG_FILE_1 = "/home/paul/data_quiche/qlog/server_fixed/client-ad6e602ee063290ca65374a3b958899ed1706754.sqlog"
# QLOG_FILE_1 = "/home/paul/data_quiche/qlog/original_pathfixed_100MB_bw100-5_dl200-5/client-3a45299803d32c76c36748f9f1742b652aa03da2.sqlog"
QLOG_FILE_1 = "/tmp/minitopo_experiences/client-386bc8df46feb429535470f229b6c20a91a0cf5c.sqlog"
# QLOG_FILE_1 = "/home/paul/data_quiche/qlog/original_pathfixed_100MB_bw100-5_dl200-5/client-0b935f79ce8418862e41f05c79dcbaa6f1d3f203.sqlog"
LABEL_1 = "client"
# QLOG_FILE_2 = "/home/paul/data_quiche/qlog/dev/client-b6c980769c4b7147c4db43b85efbeac6212029a0.sqlog"
# QLOG_FILE_2 = "/home/paul/data_quiche/qlog/server_fixed/server-4520145433d41f900d3b8d4be0235b413ace4031.sqlog"
# QLOG_FILE_2 = "/home/paul/data_quiche/qlog/original_pathfixed_100MB_bw100-5_dl200-5/server-1493c9f706e0c8e40c22d876b8a87a86af4d2adb.sqlog"
QLOG_FILE_2 = "/tmp/minitopo_experiences/server-32d3cdaa48e3c30f7771adbe9c7180b229855430.sqlog"
# QLOG_FILE_2 = "/home/paul/data_quiche/qlog/original_pathfixed_100MB_bw100-5_dl200-5/server-3d49528e79d439eaab7832e275d3c13cf1ec6455.sqlog"
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
                if time is None or path_id is None or latest_rtt is None or bif is None or cwnd is None:
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
    # compute headroom ratio when possible
    if "cwnd" in df.columns:
        df["headroom_ratio"] = (df["cwnd"] - df["bytes_in_flight"]) / df["cwnd"]
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

    # 1) Headroom Ratio
    if PLOT_HEADROOM:
        plt.figure()
        plt.plot(df1["time_s"], df1["headroom_ratio"], label=label1)
        plt.plot(df2["time_s"], df2["headroom_ratio"], label=label2)
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
        plt.plot(df1["time_s"], df1["rtt_variance"], label=label1)
        plt.plot(df2["time_s"], df2["rtt_variance"], label=label2)
        plt.title("RTT Variance Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("RTT Variance (ms^2)")
        plt.legend()

        # 4) Bytes In Flight
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
        for pid in sorted(df1["path_id"].unique()):
            d = df1[df1["path_id"] == pid]
            plt.plot(d["time_s"], d["cwnd"], label=f"{label1}-path{pid}")
        for pid in sorted(df2["path_id"].unique()):
            d = df2[df2["path_id"] == pid]
            plt.plot(d["time_s"], d["cwnd"], label=f"{label2}-path{pid}")
        plt.title("Congestion Window Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Congestion Window (bytes)")
        plt.legend()
    elif PLOT_CWND:
        print("Warning: 'cwnd' column not found; skipping CWND plot.")
    
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
    if PLOT_PACING_RATE and "pacing_rate" in df2.columns:
        plt.figure()
        # for pid in sorted(df1["path_id"].unique()):
        #     d = df1[df1["path_id"] == pid]
        #     plt.plot(d["time_s"], d["pacing_rate"], label=f"{label1}-path{pid}")
        for pid in sorted(df2["path_id"].unique()):
            d = df2[df2["path_id"] == pid]
            plt.plot(d["time_s"], d["pacing_rate"], label=f"{label2}-path{pid}")
        plt.title("Pacing Rate Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Pacing Rate (bytes/s)")
        plt.legend()

    plt.show()


if __name__ == "__main__":
        # 1) Compute payload distribution stats
    if CALC_PAYLOAD_STATS:
        dfp1 = parse_payload_qlog(QLOG_FILE_1)
        dfp2 = parse_payload_qlog(QLOG_FILE_2)
        compute_payload_stats(dfp1, LABEL_1)
        compute_payload_stats(dfp2, LABEL_2)

    plot_trends(QLOG_FILE_1, LABEL_1, QLOG_FILE_2, LABEL_2)