import json

def truncate_qlog_by_time(input_file_path, output_file_path, start_time_ms, end_time_ms):
    """
    Truncates a qlog file to include entries within a specified time range.

    Args:
        input_file_path (str): The path to the original qlog file.
        output_file_path (str): The path where the truncated qlog will be saved.
        start_time_ms (float): The start timestamp in milliseconds (inclusive).
        end_time_ms (float): The end timestamp in milliseconds (inclusive).
    """
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        # qlog files can start with a header that's not a JSON object,
        # or they might be a single JSON array with all events.
        # The sample you provided suggests newline-delimited JSON objects,
        # often preceded by a non-JSON character like ''.
        # We'll try to handle both common formats.

        # Try to read the first character. If it's not '{', it might be a prefix.
        # Then try to read as a single JSON array (common for browsers).
        # Otherwise, assume newline-delimited JSON.

        first_char = infile.read(1)
        infile.seek(0) # Go back to the beginning of the file

        if first_char == '[':
            # It's likely a single JSON array. Load it entirely.
            try:
                full_qlog_data = json.load(infile)
                if isinstance(full_qlog_data, list):
                    # Find the "events" key if it's a standard qlog format
                    events = []
                    for item in full_qlog_data:
                        if isinstance(item, dict) and "events" in item:
                            events.extend(item["events"])
                        # Handle cases where the top-level is just a list of events
                        elif isinstance(item, list) and all(isinstance(e, list) and len(e) >= 3 for e in item):
                            # This heuristic checks for the old qlog format [time, name, data]
                            events.extend(item)
                        elif isinstance(item, dict) and "time" in item:
                            # It could be a list of direct event objects (less common for full files)
                            events.append(item)


                    truncated_events = []
                    for event_entry in events:
                        if isinstance(event_entry, list) and len(event_entry) > 0:
                            # Old qlog format: [timestamp, event_type, event_data]
                            timestamp = event_entry[0]
                        elif isinstance(event_entry, dict) and "time" in event_entry:
                            # New qlog format: {"time": ..., "name": ..., "data": ...}
                            timestamp = event_entry["time"]
                        else:
                            continue # Skip malformed entries

                        if start_time_ms <= timestamp <= end_time_ms:
                            truncated_events.append(event_entry)
                    
                    # For full qlog files, we might need to reconstruct the structure
                    # This is a simplification. For truly robust handling of full qlog files,
                    # you'd need to parse the entire JSON structure and insert events back.
                    # For now, we'll just write the filtered events as a new array.
                    json.dump(truncated_events, outfile, indent=2)
                    return

            except json.JSONDecodeError:
                infile.seek(0) # Reset if it wasn't a single JSON array


        # Assume newline-delimited JSON objects, possibly with a prefix
        for line in infile:
            # Remove any non-JSON prefix characters (like '')
            stripped_line = line.lstrip(' \t\n\r\f\v\x1e') # \x1e is the record separator
            if not stripped_line:
                continue

            try:
                # Parse the JSON object
                entry = json.loads(stripped_line)

                # Check if "time" key exists and is within the range
                if "time" in entry and start_time_ms <= entry["time"] <= end_time_ms:
                    # Write the entry to the output file, followed by a newline and the prefix
                    # We re-add the '' if it was present in the original sample,
                    # or just write as plain JSON lines if it wasn't universally present.
                    # For simplicity, we'll write plain JSON lines unless the user confirms
                    # that the '' is strictly part of the format.
                    # Based on the sample, it seems like each line is prefixed.
                    outfile.write(line.split('{', 1)[0] + json.dumps(entry) + '\n')

            except json.JSONDecodeError as e:
                # print(f"Skipping malformed line: {line.strip()} - Error: {e}")
                # This could be a header line or other non-JSON content.
                # If you need to preserve the qlog header, you'd parse it separately
                # before entering this loop.
                pass

# Example usage:
input_qlog_file = "/home/paul/data_quiche/qlog/experiments_0610-14/fc_updated-rttback/server-separate-fc_updated-rttback-0loss-BBR .txt"  # Replace with your input file name
output_qlog_file = "/home/paul/data_quiche/qlog/experiments_0610-14/fc_updated-rttback/truncated50006500server.txt"
start_timestamp = 5000.0  # in milliseconds
end_timestamp = 6500.0    # in milliseconds

print(f"Truncating qlog from {start_timestamp}ms to {end_timestamp}ms...")
truncate_qlog_by_time(input_qlog_file, output_qlog_file, start_timestamp, end_timestamp)
print(f"Truncation complete. Output saved to {output_qlog_file}")

# You can then read the output_qlog_file to verify
with open(output_qlog_file, 'r') as f:
    print("\nContent of truncated_qlog.qlog:")
    print(f.read())