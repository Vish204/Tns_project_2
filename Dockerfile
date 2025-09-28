FROM python:3.11-slim

# Saner defaults and local fallback
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# System build deps (clean layer afterwards)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        pkg-config \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# For documentation; routing still depends on what the process binds to
EXPOSE 8501

# Use a shell so $PORT expands at runtime (Render sets PORT for web services)
CMD bash -lc "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"



































# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt /app/

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gfortran \
#     libopenblas-dev \
#     pkg-config \
#     ninja-build

# RUN pip install --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . /app

# EXPOSE 8501

# CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
 

#for dependent of app.py and main.py
#comment out only once 
# # FROM python:3.12-slim

# # WORKDIR /app

# # COPY requirements.txt /app/''

# # # Install build tools for pip to compile C extensions
# # # RUN apt-get update && apt-get install -y build-essential 2nd error
# # #RUN apt-get update && apt-get install -y build-essential gfortran 3rd error
# # RUN apt-get update && apt-get install -y \
# #     build-essential \
# #     gfortran \
# #     libopenblas-dev \
# #     pkg-config

# # RUN pip install --no-cache-dir -r requirements.txt

# # COPY . /app

# # EXPOSE 8000 8501

# # COPY start.sh /app/start.sh
# # RUN chmod +x /app/start.sh

# # CMD ["/app/start.sh"]


# #5th error
# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt /app/

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gfortran \
#     libopenblas-dev \
#     pkg-config \
#     ninja-build

# RUN pip install --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . /app

# EXPOSE 8000 8501

# COPY start.sh /app/start.sh
# RUN chmod +x /app/start.sh

# CMD ["/app/start.sh"]
