FROM python:3.12-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ARG TORCH_VERSION=2.13.0

RUN python -m venv "${VIRTUAL_ENV}"
WORKDIR /build

COPY pyproject.toml ./
# Hatch needs the declared package to resolve project metadata. Copying only
# this stable marker keeps the expensive ML dependency layer cacheable while
# application code changes during normal development.
COPY src/imbalance_pipeline/__init__.py ./src/imbalance_pipeline/__init__.py
COPY infra/clickhouse ./infra/clickhouse

# The serving image is CPU-only. Installing PyTorch from its CPU index avoids
# pulling several gigabytes of CUDA libraries into every worker.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir ".[ml]"

COPY src ./src
COPY infra ./infra
RUN pip install --no-cache-dir --no-deps .

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 imbalance \
    && useradd --uid 10001 --gid imbalance --create-home --home-dir /app imbalance

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=imbalance:imbalance src ./src
COPY --chown=imbalance:imbalance infra ./infra
COPY --chown=imbalance:imbalance pyproject.toml ./

USER 10001:10001
EXPOSE 8000
