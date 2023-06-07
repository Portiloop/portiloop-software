#!/bin/bash

output_file="config_files.txt"

# List of files to concatenate
file_list=("/etc/asound.conf" "/etc/systemd/system/create_ap.service" "/etc/systemd/system/jupyter.service" "/etc/systemd/system/setup_tables.service" "/usr/local/bin/create_ap0.sh" "/etc/hostapd/hostapd.conf" "/etc/dnsmasq.conf" "/usr/local/bin/setup_tables.sh")

# Create an empty output file
echo -n > "$output_file"

# Loop through each file
for file_name in "${file_list[@]}"; do
    echo "File: $(basename "$file_name")" >> "$output_file"  # Add file name
    echo "---------------------------------------------------" >> "$output_file"
    cat "$file_name" >> "$output_file"    # Concatenate file content
    echo >> "$output_file"
done

echo "Concatenation completed. Output file: $output_file"
