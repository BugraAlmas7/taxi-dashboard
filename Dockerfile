FROM python:3.12-slim
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y \
    libgomp1 tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
# torch comes from PyPI (no CUDA index): the plain wheel works on BOTH x86_64 and
# ARM64 (Apple Silicon Macs run linux/arm64 containers), everything runs on CPU.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
COPY . .
EXPOSE 8000
# Invoked via sh so a missing exec-bit on the mounted host file doesn't matter.
ENTRYPOINT ["sh", "/app/entrypoint.sh"]