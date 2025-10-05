FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 라이브러리 설치
# - curl: healthcheck용
# - libgomp1: LightGBM 실행에 필요한 OpenMP 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 (캐시 최적화)
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY open_aq.py .
COPY config.py .

# 비루트 사용자 생성 및 권한 설정
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 애플리케이션 실행
CMD ["uvicorn", "open_aq:app", "--host", "0.0.0.0", "--port", "8000"]
