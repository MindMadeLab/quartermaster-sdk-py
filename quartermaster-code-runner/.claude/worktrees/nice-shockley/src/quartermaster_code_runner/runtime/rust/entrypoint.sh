#!/bin/sh
set -e

chown runner:runner /tmp
chown runner:runner /metadata 2>/dev/null || true
chown runner:runner /workspace 2>/dev/null || true

exec su --preserve-environment -s /bin/sh -c '
    set -e
    cd /workspace
    export HOME=/workspace
    export CARGO_HOME=/workspace/.cargo
    mkdir -p $CARGO_HOME

    # Decode code to default filename
    if [ -n "$ENCODED_CODE" ]; then
        echo "$ENCODED_CODE" | base64 -d > main.rs
    fi

    # Extract additional files if provided
    if [ -n "$ENCODED_FILES" ]; then
        echo "$ENCODED_FILES" | base64 -d | tar -xz
    fi

    # Copy default Cargo.toml if user did not provide one
    if [ ! -f Cargo.toml ]; then
        cp /app/Cargo.toml.default Cargo.toml
    fi

    # Execute custom entrypoint or default
    if [ -n "$CUSTOM_ENTRYPOINT" ]; then
        exec sh -c "$CUSTOM_ENTRYPOINT"
    else
        mkdir -p src
        mv main.rs src/main.rs
        exec cargo run --release -q
    fi
' runner
