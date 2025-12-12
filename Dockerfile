FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
COPY .python-version .

RUN uv sync

COPY ai_text_detector ai_text_detector
COPY commands.py .

ENTRYPOINT ["uv", "run", "commands.py"]
