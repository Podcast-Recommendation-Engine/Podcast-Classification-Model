# Use Python 3.13 base image
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install .
COPY src/ src/

CMD ["python", "src/main.py"]
