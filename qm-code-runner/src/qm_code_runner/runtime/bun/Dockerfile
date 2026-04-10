FROM oven/bun:1-slim

LABEL qm.name="Bun"
LABEL qm.description="Bun runtime with TypeScript support"
LABEL qm.default_entrypoint="bun main.ts"
LABEL qm.file_extension=".ts"
LABEL qm.main_file="main.ts"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tar && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1001 --no-create-home --shell /bin/sh runner && \
    mkdir -p /home/runner/.bun && \
    chown -R runner:runner /home/runner

COPY entrypoint.sh .
COPY sdk.ts .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
