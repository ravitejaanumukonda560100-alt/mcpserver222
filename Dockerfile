FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600
ENV DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE=30

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "pytest", "tests/pytest_tests/test_mcp.py", "-vv", "-s"]