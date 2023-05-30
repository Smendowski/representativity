FROM python:3.10-slim-bullseye AS base
ENV PYTHONUNBUFFERED=true
WORKDIR /app

FROM base AS builder
ENV POETRY_VERSION=1.5
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN python -m pip install "poetry==$POETRY_VERSION"
ADD data ./data
ADD ml ./ml
COPY main.py services.py logs.py exceptions.py poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv

FROM base AS tester
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=builder /app /app
COPY conftest.py ./
COPY data ./data
ADD tests ./tests
CMD ["pytest"]

FROM base AS runner
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=builder /app /app
COPY .env ./.env
CMD env $(cat .env | grep -v '^#' | xargs)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
