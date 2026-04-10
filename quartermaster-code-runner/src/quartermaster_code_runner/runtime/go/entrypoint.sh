#!/bin/sh
set -e

chown runner:runner /tmp
chown runner:runner /metadata 2>/dev/null || true
chown runner:runner /workspace 2>/dev/null || true

exec su -p -s /bin/sh -c '
    set -e
    cd /workspace
    export HOME=/workspace
    export GOCACHE=/workspace/.cache/go-build
    export GOFLAGS="-buildvcs=false"
    mkdir -p $GOCACHE

    # Copy SDK for user code access and create go module structure
    mkdir -p /workspace/sdk
    cp /app/sdk.go /workspace/sdk/sdk.go

    # Decode code to default filename
    if [ -n "$ENCODED_CODE" ]; then
        echo "$ENCODED_CODE" | base64 -d > main.go
    fi

    # Extract additional files if provided
    if [ -n "$ENCODED_FILES" ]; then
        echo "$ENCODED_FILES" | base64 -d | tar -xz
    fi

    # Create go.mod for the user code with local SDK
    cat > go.mod << EOF
module main
go 1.22
require sdk v0.0.0
replace sdk => ./sdk
EOF

    # Create go.mod for the SDK
    cat > /workspace/sdk/go.mod << EOF
module sdk
go 1.22
EOF

    # Execute custom entrypoint or default
    if [ -n "$CUSTOM_ENTRYPOINT" ]; then
        exec sh -c "$CUSTOM_ENTRYPOINT"
    else
        exec go run main.go
    fi
' runner
