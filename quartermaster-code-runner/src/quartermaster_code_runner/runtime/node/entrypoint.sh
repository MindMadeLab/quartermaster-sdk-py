#!/bin/sh
set -e

chown runner:runner /tmp
chown runner:runner /metadata 2>/dev/null || true

exec su -p -s /bin/sh -c '
    set -e
    cd /tmp
    export HOME=/home/runner
    export NPM_CONFIG_CACHE=/tmp/.npm

    # Copy SDK and MCP client for user code access
    cp /app/sdk.js /tmp/sdk.js
    cp /app/mcp-client.js /tmp/mcp-client.js

    # Decode code to default filename
    if [ -n "$ENCODED_CODE" ]; then
        echo "$ENCODED_CODE" | base64 -d > main.js
    fi

    # Extract additional files if provided
    if [ -n "$ENCODED_FILES" ]; then
        echo "$ENCODED_FILES" | base64 -d | tar -xz
    fi

    # Execute custom entrypoint or default
    if [ -n "$CUSTOM_ENTRYPOINT" ]; then
        exec sh -c "$CUSTOM_ENTRYPOINT"
    else
        exec node main.js
    fi
' runner
