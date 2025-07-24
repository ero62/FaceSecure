# Python tabanlı yüz tanıma projesi için Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem paketleri
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kodları ve modeli kopyala
COPY . .

# Bağımlılıkları yükle
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Varsayılan olarak Streamlit başlat
CMD ["streamlit", "run", "app.py"]

# API başlatmak için (manuel override):
# docker run ... python -m api.main 