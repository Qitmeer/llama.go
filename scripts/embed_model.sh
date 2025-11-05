#!/usr/bin/env bash

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <executable> <model_file>"
    echo "Example: $0 build/bin/modelembed cmd/modelembed/models/Qwen3-8B-Q4_K_M.gguf"
    exit 1
fi

EXECUTABLE="$1"
MODEL_FILE="$2"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found: $EXECUTABLE"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi

# Get file sizes
EXEC_SIZE=$(stat -f %z "$EXECUTABLE" 2>/dev/null || stat -c %s "$EXECUTABLE")
MODEL_SIZE=$(stat -f %z "$MODEL_FILE" 2>/dev/null || stat -c %s "$MODEL_FILE")

echo "Embedding model into executable..."
echo "  Executable: $EXECUTABLE ($EXEC_SIZE bytes)"
echo "  Model: $MODEL_FILE ($(echo "scale=2; $MODEL_SIZE / 1024 / 1024" | bc) MB)"

# Create backup
BACKUP="${EXECUTABLE}.backup"
cp "$EXECUTABLE" "$BACKUP"
echo "  Created backup: $BACKUP"

# Create temporary file for the combined binary
TEMP_FILE="${EXECUTABLE}.tmp"

# Combine executable and model
cat "$EXECUTABLE" "$MODEL_FILE" > "$TEMP_FILE"

# Append metadata (offset and size in little-endian format)
# Offset is the original executable size
# Size is the model file size
# Use Python if available, otherwise use xxd
if command -v python3 &> /dev/null; then
    python3 -c "import struct; import sys; sys.stdout.buffer.write(struct.pack('<QQ', $EXEC_SIZE, $MODEL_SIZE))" >> "$TEMP_FILE"
elif command -v python &> /dev/null; then
    python -c "import struct; import sys; sys.stdout.write(struct.pack('<QQ', $EXEC_SIZE, $MODEL_SIZE))" >> "$TEMP_FILE"
else
    # Fallback to printf and xxd (works on macOS and Linux)
    # Convert to hex, reverse byte order for little-endian
    printf "%016x%016x" "$EXEC_SIZE" "$MODEL_SIZE" | \
        sed 's/\(..\)/\1\n/g' | \
        tac | \
        head -8 | tr -d '\n' | xxd -r -p >> "$TEMP_FILE"
    printf "%016x%016x" "$EXEC_SIZE" "$MODEL_SIZE" | \
        sed 's/\(..\)/\1\n/g' | \
        tac | \
        tail -8 | tr -d '\n' | xxd -r -p >> "$TEMP_FILE"
fi

# Replace the original executable
mv "$TEMP_FILE" "$EXECUTABLE"

# Make it executable
chmod +x "$EXECUTABLE"

# Calculate final size
FINAL_SIZE=$(stat -f %z "$EXECUTABLE" 2>/dev/null || stat -c %s "$EXECUTABLE")
echo "Done! Final executable size: $(echo "scale=2; $FINAL_SIZE / 1024 / 1024" | bc) MB"
echo "Model embedded at offset $EXEC_SIZE, size $MODEL_SIZE bytes"