# Build context: repo root. Produced image runs as the bench-runner Job
# entry point, orchestrating deploy-wait-bench-teardown for both runtimes
# with results written to a mounted PVC.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    KUBECTL_VERSION=v1.32.0 \
    TZ=UTC

# kubectl + a couple of ergonomic utilities. No curl in final image:
# we download kubectl during build and then uninstall curl so the final
# layer is as small as we can reasonably make it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl bash coreutils \
 && curl -fsSL -o /usr/local/bin/kubectl \
       "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && chmod +x /usr/local/bin/kubectl \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (cache friendly).
COPY harness/requirements.txt /app/harness/requirements.txt
RUN pip install --no-cache-dir -r /app/harness/requirements.txt

# Repo payload.
COPY harness /app/harness
COPY manifests /app/manifests
COPY bench.sh /app/bench.sh
COPY Makefile /app/Makefile
RUN chmod +x /app/bench.sh

# Non-root user so we don't run kubectl as root inside the cluster.
RUN useradd --create-home --home-dir /home/bench --shell /bin/bash --uid 1001 bench \
 && chown -R bench:bench /app

USER bench

ENTRYPOINT ["/app/bench.sh"]
CMD ["full"]
