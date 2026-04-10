#!/bin/sh
set -e

chown runner:runner /tmp
chown runner:runner /metadata 2>/dev/null || true

exec su -p -s /bin/sh -c '
    set -e
    cd /tmp
    export HOME=/home/runner
    export DENO_DIR=/home/runner/.cache/deno

    # Copy SDK for user code access
    cp /app/sdk.ts /tmp/sdk.ts

    # Decode code to default filename
    if [ -n "$ENCODED_CODE" ]; then
        echo "$ENCODED_CODE" | base64 -d > main.ts
    fi

    # Extract additional files if provided
    if [ -n "$ENCODED_FILES" ]; then
        echo "$ENCODED_FILES" | base64 -d | tar -xz
    fi

    # Execute custom entrypoint or default (allow all permissions for simplicity)
    if [ -n "$CUSTOM_ENTRYPOINT" ]; then
        exec sh -c "$CUSTOM_ENTRYPOINT"
    else
        exec deno run --allow-all main.ts
    fi
' runner
