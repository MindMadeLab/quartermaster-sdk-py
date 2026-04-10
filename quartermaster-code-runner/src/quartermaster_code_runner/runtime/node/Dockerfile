FROM node:22-slim

LABEL qm.name="Node.js"
LABEL qm.description="Node.js 22 with npm, npx, and bun"
LABEL qm.default_entrypoint="node main.js"
LABEL qm.file_extension=".js"
LABEL qm.main_file="main.js"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tar curl unzip git && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1001 --no-create-home --shell /bin/sh runner && \
    mkdir -p /home/runner/.npm /home/runner/.bun && \
    chown -R runner:runner /home/runner

# Install bun globally accessible
ENV BUN_INSTALL=/usr/local
RUN curl -fsSL https://bun.sh/install | bash

# Pre-install common packages globally for faster startup
RUN npm install -g typescript ts-node
RUN npm install -g @gongrzhe/server-gmail-autoauth-mcp

COPY entrypoint.sh .
COPY sdk.js .
COPY mcp-client.js .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
