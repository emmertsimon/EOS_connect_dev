# Wie ohAnds "EOS connect develop"-Add-on, aber mit dem Quellstand aus ./src
# (develop + PR #368) und den Paketen direkt aus dessen requirements.txt.
FROM ghcr.io/home-assistant/base-python:3.13-alpine3.22

LABEL \
    io.hass.type="addon" \
    io.hass.name="EOS connect develop + Heizung (PR 368)" \
    io.hass.url="https://github.com/ohAnd/EOS_connect/pull/368"

RUN apk add --no-cache python3 py3-pip curl gcc python3-dev musl-dev linux-headers

COPY ./src/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && apk del --purge gcc python3-dev musl-dev linux-headers \
    && rm -rf /var/cache/apk/* /tmp/requirements.txt

# PuLPs mitgelieferter x86_64-CBC ist glibc-gelinkt und läuft nicht auf Alpine/musl.
# Daher wie im offiziellen Add-on den statischen COIN-OR-Build (Prüfsumme identisch).
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      curl -fsSL --retry 6 --retry-delay 5 --retry-all-errors --retry-max-time 120 \
        -o /tmp/cbc.tgz \
        "https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.13/Cbc-releases.2.10.13-x86_64-ubuntu22-gcc1140-static.tar.gz" && \
      echo "86d52bb5634203826f213a09cba83394623a1c44eecc75caed2a553a9ac642fa  /tmp/cbc.tgz" | sha256sum -c - && \
      tar -xzf /tmp/cbc.tgz -C /usr/local ./bin/cbc && \
      rm /tmp/cbc.tgz && \
      /usr/local/bin/cbc -quit ; \
    fi

COPY ./src /app
WORKDIR /app/src

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/ || exit 1

EXPOSE 8081
CMD ["/opt/venv/bin/python", "/app/src/eos_connect.py"]
