FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN pip3 install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
