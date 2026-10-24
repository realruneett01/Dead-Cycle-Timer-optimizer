# Profile is auto-selected at start-up from the container's CPU/memory limits (see deploy/profiles.py).
# Build slim headless image for edge boxes:  docker build --build-arg DASHBOARD=0 -t dcto:edge .
FROM python:3.12-slim

ARG DASHBOARD=1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    DCTO_IN_CONTAINER=1 DCTO_DATA_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN if [ "$DASHBOARD" = "0" ]; then \
      grep -viE '^(streamlit|plotly|matplotlib|pytest)' requirements.txt > req.txt; \
    else grep -viE '^pytest' requirements.txt > req.txt; fi \
    && pip install -r req.txt && rm req.txt

COPY simulator ./simulator
COPY opcua ./opcua
COPY detector ./detector
COPY dashboard ./dashboard
COPY deploy ./deploy

RUN useradd --system --uid 10001 dcto && mkdir /data && chown dcto /data
USER dcto
VOLUME /data
EXPOSE 8501
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s \
  CMD python -m deploy health || exit 1

CMD ["python", "-m", "deploy", "run"]
