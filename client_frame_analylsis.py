import json
from collections import defaultdict
import pandas as pd
import argparse
import sys

def analyze_qlog_data(qlog_file_path):
    """
    Analyzes qlog data for packet_sent events and frame type distribution.

    Args:
        qlog_file_path (str): Path to the qlog file.

    Returns:
        tuple: Contains overall stats, frame counts, individual frame occurrences,
               the raw packet_payload_associated_with_frame_type dictionary,
               and summary DataFrames.
    """
    packet_payload_associated_with_frame_type = defaultdict(int)
    packets_containing_frame_type = defaultdict(int)
    individual_frame_occurrences = defaultdict(int)

    total_packets_sent = 0
    total_payload_bytes_sent = 0
    processed_lines = 0
    error_lines = 0

    print(f"Processing qlog file: {qlog_file_path}", file=sys.stderr)

    with open(qlog_file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            processed_lines += 1
            try:
                line_content = line.strip()
                if line_content.endswith(','):
                    line_content = line_content[:-1]
                
                if not line_content:
                    continue
                    
                record = json.loads(line_content)

                if record.get("name") == "transport:packet_sent":
                    total_packets_sent += 1
                    packet_data = record.get("data", {})
                    raw_data = packet_data.get("raw", {})
                    payload_length = raw_data.get("payload_length", 0)
                    total_payload_bytes_sent += payload_length

                    frames = packet_data.get("frames", [])
                    if not frames:
                        # This case might not be typical for packet_sent but handled
                        packets_containing_frame_type["NO_FRAMES_IN_PACKET"] += 1
                        continue

                    unique_frame_types_in_this_packet = set()
                    for frame in frames:
                        frame_type = frame.get("frame_type")
                        if frame_type:
                            individual_frame_occurrences[frame_type] += 1
                            unique_frame_types_in_this_packet.add(frame_type)
                    
                    for f_type in unique_frame_types_in_this_packet:
                        packets_containing_frame_type[f_type] += 1
                        packet_payload_associated_with_frame_type[f_type] += payload_length
            
            except json.JSONDecodeError as e:
                error_lines += 1
                # print(f"Warning: JSON decode error on line {line_number}: {e} - Line: '{line.strip()[:100]}...'", file=sys.stderr)
            except Exception as e:
                error_lines +=1
                # print(f"Warning: An unexpected error occurred on line {line_number}: {e} - Line: '{line.strip()[:100]}...'", file=sys.stderr)

    print(f"Finished processing. Total lines: {processed_lines}, Errored lines: {error_lines}", file=sys.stderr)

    # Prepare data for DataFrame
    df_data = []
    for f_type in sorted(packets_containing_frame_type.keys(), key=lambda k: packets_containing_frame_type[k], reverse=True):
        payload_sum = packet_payload_associated_with_frame_type[f_type]
        num_packets = packets_containing_frame_type[f_type]
        avg_payload_per_packet_containing_type = payload_sum / num_packets if num_packets > 0 else 0
        percentage_of_total_payload = (payload_sum / total_payload_bytes_sent) * 100 if total_payload_bytes_sent > 0 else 0
        percentage_of_total_packets = (num_packets / total_packets_sent) * 100 if total_packets_sent > 0 else 0
        
        latex_frame_type = f_type.replace("_", r"\_")

        df_data.append({
            "Frame Type": latex_frame_type,
            "Packets Containing Type": num_packets,
            "% of Total Packets": f"{percentage_of_total_packets:.2f}\%",
            "Total Payload (bytes)": payload_sum,
            "% of Total Payload": f"{percentage_of_total_payload:.2f}\%",
            "Avg Payload/Pkt (bytes)": f"{avg_payload_per_packet_containing_type:.2f}"
        })

    df_summary = pd.DataFrame(df_data)
    
    individual_frames_data = []
    total_individual_frames = sum(individual_frame_occurrences.values())
    for f_type, count in sorted(individual_frame_occurrences.items(), key=lambda item: item[1], reverse=True):
        percentage_of_total_individual_frames = (count / total_individual_frames) * 100 if total_individual_frames > 0 else 0
        individual_frames_data.append({
            "Frame Type": f_type.replace("_", r"\_"),
            "Occurrences": count,
            "% of All Frames": f"{percentage_of_total_individual_frames:.2f}\%"
        })
    df_individual_frames = pd.DataFrame(individual_frames_data)

    return (total_packets_sent, total_payload_bytes_sent, 
            packets_containing_frame_type, individual_frame_occurrences,
            packet_payload_associated_with_frame_type, # Added this
            df_summary, df_individual_frames)

def main(qlog_file_path):
    (total_packets_sent, total_payload_bytes_sent,
     packets_containing_frame_type, individual_frame_occurrences,
     packet_payload_associated_with_frame_type, # Added this
     df_summary, df_individual_frames) = analyze_qlog_data(qlog_file_path)

    print("\n--- Overall Statistics ---")
    print(f"Total 'transport:packet_sent' events (packets): {total_packets_sent}")
    print(f"Total payload bytes sent across all packets: {total_payload_bytes_sent} bytes")

    print("\n--- LaTeX: Individual Frame Occurrences ---")
    if not df_individual_frames.empty:
        latex_individual_frames = df_individual_frames.to_latex(
            index=False,
            escape=False, 
            caption="Distribution of Individual Frame Types in Sent Packets.",
            label="tab:individual_frame_distribution",
            column_format="lrr", 
            header=["Frame Type", "Occurrences", r"\% of All Frames"]
        )
        # Simple booktabs-like structure
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(latex_individual_frames.replace(r'\toprule', r'\toprule' + '\n' + r'\multicolumn{3}{c}{Individual Frame Occurrences} \\' + '\n' + r'\midrule')
                                     .replace(r'\midrule', r'\midrule')
                                     .replace(r'\bottomrule', r'\bottomrule'))
        print(r"\end{table}")
    else:
        print("No individual frame data to display.")


    print("\n--- LaTeX: Packet & Payload Analysis by Frame Type (Sorted by Packet Count) ---")
    if not df_summary.empty:
        df_summary_sorted_by_packet_count = df_summary.sort_values(by="Packets Containing Type", ascending=False)
        df_summary_latex = df_summary_sorted_by_packet_count.rename(columns={
            "% of Total Packets": r"\% of Total Pkts",
            "Total Payload (bytes)": "Payload in Pkts (B)", 
            "% of Total Payload": r"\% of Total Payload",
            "Avg Payload/Pkt (bytes)": "Avg Payload/Pkt (B)"
        })
        latex_summary_by_packet_count = df_summary_latex.to_latex(
            index=False,
            escape=False, 
            caption="Packet and Payload Analysis by Frame Type (Sorted by Packet Count). A single packet can contain multiple frame types.",
            label="tab:frame_type_packet_payload_analysis",
            column_format="lrrrrr", 
            header=["Frame Type", "Pkts w/ Type", r"\% of Total Pkts", "Payload in Pkts (B)", r"\% of Total Payload", "Avg Payload/Pkt (B)"]
        )
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(latex_summary_by_packet_count.replace(r'\toprule', r'\toprule' + '\n' + r'\multicolumn{6}{c}{Packet \& Payload Analysis by Frame Type} \\' + '\n' + r'\midrule')
                                         .replace(r'\midrule', r'\midrule')
                                         .replace(r'\bottomrule', r'\bottomrule'))
        print(r"\end{table}")
    else:
        print("No summary data to display.")


    # Specifics for mp_ack
    mp_ack_key_orig = "mp_ack"
    
    if mp_ack_key_orig in packets_containing_frame_type:
        mp_ack_individual_frames = individual_frame_occurrences.get(mp_ack_key_orig, 0)
        mp_ack_packets = packets_containing_frame_type.get(mp_ack_key_orig, 0)
        mp_ack_payload = packet_payload_associated_with_frame_type.get(mp_ack_key_orig, 0)
            
        print("\n--- MP_ACK Specifics (Raw Data) ---")
        print(f"Number of individual MP_ACK frames: {mp_ack_individual_frames}")
        print(f"Number of packets containing at least one MP_ACK frame: {mp_ack_packets}")
        if total_packets_sent > 0:
            print(f"Percentage of total packets containing MP_ACK: {(mp_ack_packets / total_packets_sent) * 100:.2f}%")
        print(f"Total payload (bytes) of packets containing MP_ACK: {mp_ack_payload}")
        if total_payload_bytes_sent > 0:
            print(f"Percentage of total payload bytes from packets containing MP_ACK: {(mp_ack_payload / total_payload_bytes_sent) * 100:.2f}%")
        if mp_ack_packets > 0:
            print(f"Average payload of a packet containing MP_ACK: {mp_ack_payload / mp_ack_packets:.2f} bytes")
    else:
        print("\n--- MP_ACK Specifics ---")
        print("No MP_ACK frames found in the log.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze QUIC qlog files for frame type distribution.")
    parser.add_argument("qlog_file", help="Path to the qlog file (.qlog or .sqlog)")
    
    args = parser.parse_args()
    main(args.qlog_file)