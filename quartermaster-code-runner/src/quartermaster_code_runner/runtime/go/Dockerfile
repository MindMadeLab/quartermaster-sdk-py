FROM golang:1.22-alpine

LABEL qm.name="Go"
LABEL qm.description="Go 1.22 runtime"
LABEL qm.default_entrypoint="go run main.go"
LABEL qm.file_extension=".go"
LABEL qm.main_file="main.go"

WORKDIR /app

RUN apk add --no-cache tar && \
    adduser -D -u 1001 -s /bin/sh runner

COPY entrypoint.sh .
COPY sdk.go .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
