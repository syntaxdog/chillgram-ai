FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pymatting의 numba cache 강제 사용 때문에 컨테이너에서 import 단계 크래시 발생 -> cache 비활성화 패치
RUN sed -i 's/cache=True/cache=False/g' \
  /usr/local/lib/python3.11/site-packages/pymatting/util/kdtree.py

COPY . .

ENV PYTHONUNBUFFERED=1
ENV NUMBA_DISABLE_CACHE=1

CMD ["python", "rabbit_worker.py"]
