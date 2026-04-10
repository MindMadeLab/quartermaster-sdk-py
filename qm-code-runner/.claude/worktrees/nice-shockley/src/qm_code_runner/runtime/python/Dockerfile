FROM python:3.12-slim

LABEL qm.name="Python"
LABEL qm.description="Python 3.12 with uv/uvx package manager"
LABEL qm.default_entrypoint="python main.py"
LABEL qm.file_extension=".py"
LABEL qm.main_file="main.py"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tar curl git && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1001 --no-create-home --shell /bin/sh runner

# Install uv package manager for uvx support
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx && \
    chmod 755 /usr/local/bin/uv /usr/local/bin/uvx

# Make uv tool bin dir available to all users
ENV PATH="/root/.local/bin:$PATH"

# Pre-install common Python packages
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    httpx \
    beautifulsoup4 \
    openpyxl \
    google-auth \
    google-auth-oauthlib \
    google-api-python-client

COPY entrypoint.sh .
COPY sdk.py .
COPY mcp-client.py .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
