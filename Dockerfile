# ==============================================================================
# Getnet Multi-Agent Support System - Production Dockerfile
# ==============================================================================

FROM python:3.11-slim

# Evita criação de arquivos .pyc e força flush imediato nos logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Instala dependências do sistema necessárias para compilação mínima
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requisitos e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação e base de conhecimento
COPY data/ ./data/
COPY app/ ./app/
COPY tests/ ./tests/
COPY pytest.ini .
COPY .env.example .env

# Expõe a porta padrão da API
EXPOSE 8000

# Verificação de integridade do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
