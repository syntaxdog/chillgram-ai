FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ pymatting 전체에서 njit(cache=True) 제거 (kdtree만 하면 boxfilter 등에서 또 터짐)
RUN find /usr/local/lib/python3.11/site-packages/pymatting -name "*.py" -print0 \
 | xargs -0 sed -i 's/cache=True/cache=False/g'

# (선택) 빌드 단계에서 cache=True가 남았는지 검증 로그
RUN grep -R "cache=True" -n /usr/local/lib/python3.11/site-packages/pymatting || true

COPY . .

ENV PYTHONUNBUFFERED=1
ENV NUMBA_DISABLE_CACHE=1
ENV U2NET_HOME=/app/.u2net

# rembg U2Net 모델 사전 다운로드 (런타임 콜드스타트 제거)
# --user 1005:1006 실행 시에도 읽기 가능하도록 권한 설정
RUN python -c "from rembg import remove; from PIL import Image; remove(Image.new('RGBA',(1,1)))" \
    && chmod -R 755 /app/.u2net

CMD ["python", "rabbit_worker.py"]
