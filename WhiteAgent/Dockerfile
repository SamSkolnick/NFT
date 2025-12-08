FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY evaluate.py .

# Default command (can be overridden by Green Agent)
CMD ["python", "evaluate.py"]
