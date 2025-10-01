FROM python:3.13.7-slim-trixie AS base

RUN adduser --uid 1000 --disabled-password --gecos '' appuser
USER 1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

RUN pip install --user --no-cache-dir --upgrade pip

FROM base AS init
COPY ./requirements.txt /home/appuser/requirements.txt
RUN pip install --user --no-cache-dir  --upgrade -r /home/appuser/requirements.txt

FROM init AS emb
RUN python -m pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --user --no-cache-dir sentence-transformers==5.1.1

FROM init AS mcp
COPY ./requirements.mcp.txt /home/appuser/requirements.mcp.txt
RUN pip install --user --no-cache-dir --upgrade -r /home/appuser/requirements.mcp.txt
