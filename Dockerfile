FROM python:3.13.7-slim-trixie AS base

RUN adduser --uid 1000 --disabled-password --gecos '' appuser
USER 1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

RUN pip install --user --no-cache-dir --upgrade pip
COPY ./requirements.txt /home/appuser/requirements.txt
RUN pip install --user --no-cache-dir  --upgrade -r /home/appuser/requirements.txt

FROM base AS mcp
COPY ./requirements.mcp.txt /home/appuser/requirements.mcp.txt
RUN pip install --user --no-cache-dir --upgrade -r /home/appuser/requirements.mcp.txt
