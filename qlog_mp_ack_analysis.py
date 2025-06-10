#!/usr/bin/env python3
"""
qlog_mp_ack_analysis.py

Parse a single QLOG file for mp_ack frames and compute & visualize:
 1. Total mp_ack bytes for chosen ack_path (path0, path1, or all).
 2. mp_ack sending rate in bytes/sec per event.
 3. ack-data ratio: data bytes acknowledged by each mp_ack frame / ack-frame bytes.

Assumptions:
 - QLOG timestamps are in milliseconds.
 - Acknowledged data packets should be matched by both packet number and path_id.

Usage:
  python qlog_mp_ack_analysis.py --qlog <file> --path [path0|path1|all]

Produces time-series plots for each metric.
"""

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_packet_sent(path):
    """
    Map each transport:packet_sent event to its sending path_id and total stream data bytes.
    Returns a DataFrame with columns: packet_number, path_id, stream_data_bytes
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
            if ev.get('name') != 'transport:packet_sent':
                continue
            data = ev.get('data', {})
            header = data.get('header', {})
            pkt = header.get('packet_number')
            pid = data.get('path_id')
            # sum bytes of stream frames
            stream_bytes = sum(
                fr.get('length', 0)
                for fr in data.get('frames', [])
                if fr.get('frame_type') == 'stream'
            )
            if pkt is not None and pid is not None:
                records.append({
                    'packet_number': pkt,
                    'path_id': pid,
                    'stream_data_bytes': stream_bytes
                })
    return pd.DataFrame(records)


def parse_mp_ack(path):
    """
    Extract mp_ack events from transport:packet_received frames.
    Returns a DataFrame with columns:
      time (ms), recv_path, ack_path, ack_bytes, acked_ranges
    """
    recs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('name') != 'transport:packet_received':
                continue
            t = ev.get('time')  # timestamp in ms
            data = ev.get('data', {})
            raw = data.get('raw', {})
            length = raw.get('length')
            recv_pid = data.get('path_id')
            for fr in data.get('frames', []):
                ft = fr.get('frame_type', '').lower()
                if ft.startswith('mp_ack'):
                    recs.append({
                        'time': t,
                        'recv_path': recv_pid,
                        'ack_path': fr.get('path_identifier'),
                        'ack_bytes': length,
                        'acked_ranges': fr.get('acked_ranges', [])
                    })
    return pd.DataFrame(recs)


def flatten_ranges(ranges):
    """
    Flatten inclusive acked_ranges into a set of packet numbers.
    """
    s = set()
    for a, b in ranges:
        s.update(range(a, b + 1))
    return s


def main():
    parser = argparse.ArgumentParser(
        description='Analyze mp_ack frames in a QLOG file and visualize metrics.'
    )
    parser.add_argument('--qlog', '-q', required=True,
                        help='Path to a single QLOG file')
    parser.add_argument('--path', choices=['path0', 'path1', 'all'], default='path1',
                        help='Which ack_path to analyze')
    args = parser.parse_args()

    # Parse sent packets with path_id
    sent = parse_packet_sent(args.qlog)
    if sent.empty:
        print('No packet_sent events found.'); return

    # Parse mp_ack events
    ack = parse_mp_ack(args.qlog)
    if ack.empty:
        print('No mp_ack events found in QLOG.'); return

    # Filter by ack_path
    if args.path != 'all':
        pid = int(args.path[-1])
        ack = ack[ack['ack_path'] == pid]
    if ack.empty:
        print(f'No mp_ack events for {args.path}.'); return

    # Sort by time
    ack = ack.sort_values('time').reset_index(drop=True)

    # Compute newly acknowledged data bytes per event
    seen = set()
    data_list = []
    for _, row in ack.iterrows():
        newly = flatten_ranges(row['acked_ranges']) - seen
        seen |= newly
        total_data = 0
        for pkt in newly:
            rec = sent[
                (sent['packet_number'] == pkt) &
                (sent['path_id'] == row['ack_path'])
            ]
            if not rec.empty:
                total_data += rec.iloc[0]['stream_data_bytes']
        data_list.append(total_data)
    ack['acked_data_bytes'] = data_list

    # Compute time deltas (ms and sec) and sending rate
    ack['delta_t_ms'] = ack['time'].diff().fillna(0)
    ack['delta_t_s'] = ack['delta_t_ms'] / 1000.0
    ack['rate_bps'] = ack['ack_bytes'] / ack['delta_t_s'].replace({0: np.nan})
    ack['rate_bps'] = ack['rate_bps'].fillna(0)

    # Compute ack-data ratio
    ack['data_ratio'] = ack['acked_data_bytes'] / ack['ack_bytes']

    # Align time to zero for plotting (ms)
    t0 = ack['time'].iloc[0]
    ack['time_ms'] = ack['time'] - t0

    # Summary output
    print(f"Total mp_ack bytes ({args.path}): {ack['ack_bytes'].sum()} bytes")

    # Plot: ack_bytes over time
    plt.figure()
    plt.plot(ack['time_ms'], ack['ack_bytes'], 'o-')
    plt.title('mp_ack Frame Bytes over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Ack Frame Bytes')
    plt.grid(True)

    # Plot: sending rate
    plt.figure()
    plt.plot(ack['time_ms'], ack['rate_bps'], 'o-')
    plt.title('mp_ack Sending Rate')
    plt.xlabel('Time (ms)')
    plt.ylabel('Bytes/sec')
    plt.grid(True)

    # Plot: data bytes acknowledged
    plt.figure()
    plt.plot(ack['time_ms'], ack['acked_data_bytes'], 'o-')
    plt.title('Stream Data Bytes Acknowledged')
    plt.xlabel('Time (ms)')
    plt.ylabel('Data Bytes')
    plt.grid(True)

    # Plot: ack-data ratio
    plt.figure()
    plt.plot(ack['time_ms'], ack['data_ratio'], 'o-')
    plt.title('Ack-Data Ratio')
    plt.xlabel('Time (ms)')
    plt.ylabel('Data Bytes / Ack Bytes')
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    main()
