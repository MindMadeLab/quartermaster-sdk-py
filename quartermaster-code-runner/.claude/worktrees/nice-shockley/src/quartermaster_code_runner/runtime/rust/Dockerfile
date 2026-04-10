FROM rust:1.87-slim

LABEL qm.name="Rust"
LABEL qm.description="Rust 1.87 with Cargo"
LABEL qm.default_entrypoint="cargo run"
LABEL qm.file_extension=".rs"
LABEL qm.main_file="src/main.rs"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tar && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1001 --no-create-home --shell /bin/sh runner && \
    mkdir -p /home/runner/.cargo && \
    chown -R runner:runner /home/runner

# Pre-compile SDK with serde to cache dependencies
COPY sdk/ /app/sdk/
RUN cd /app/sdk && cargo build --release && \
    chown -R runner:runner /app/sdk

COPY Cargo.toml.default .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
