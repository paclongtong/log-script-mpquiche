#!/bin/bash

# --- Configuration ---
input_file="quiche_client.log" # Replace with your actual log file name
output_file="quiche_client_truncated.log"     # Name for the extracted middle part
# -------------------

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

echo "Processing file: $input_file"

# 1. Get total size in bytes
total_size=$(stat -c %s "$input_file")
if [ $? -ne 0 ] || [ -z "$total_size" ]; then
    echo "Error: Could not get size of '$input_file'."
    exit 1
fi
echo "Total size: $total_size bytes"

# Handle very small files (less than 10 bytes)
if [ "$total_size" -lt 10 ]; then
    echo "Warning: File is too small to extract a meaningful 1/10th middle chunk."
    # Decide how to handle: copy the whole file? exit?
    # cp "$input_file" "$output_file"
    # echo "Copied the whole small file."
    exit 0 # Or exit 1 if this is an error condition for you
fi

# 2. Calculate the size of the 1/10th chunk (integer division)
chunk_size=$((total_size / 10))
echo "Target chunk size: $chunk_size bytes"

# 3. Calculate the starting offset (bytes to skip)
# Formula: (total_size / 2) - (chunk_size / 2) = (total_size - chunk_size) / 2
start_offset=$(( (total_size - chunk_size) / 2 ))
echo "Starting offset (skip): $start_offset bytes"

# 4. Use dd to extract the chunk
echo "Extracting middle chunk to $output_file ..."
# Use iflag=skip_bytes and count_bytes for precision (GNU dd feature)
# bs=4k (or 1M) helps speed up the process, but skip/count are still in bytes
dd if="$input_file" of="$output_file" bs=4k skip="$start_offset" count="$chunk_size" iflag=skip_bytes,count_bytes status=progress

if [ $? -eq 0 ]; then
    echo "Successfully extracted middle chunk to $output_file"
    echo "Output file size: $(stat -c %s "$output_file") bytes"
else
    echo "Error during dd execution."
    exit 1
fi

exit 0
