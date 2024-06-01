ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY crawler .
COPY libs libs/

# Needed for pulling the git repo
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/apihome" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=crawler/requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt


# Switch to the non-privileged user to run the application.
RUN chown -R appuser:appuser /app
USER appuser
