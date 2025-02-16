FROM python:3.13-slim-bookworm


RUN mkdir -p /data
RUN chmod -R 777 /data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    zlib1g-dev \
    libssl-dev \
    python3-dev \
    python3-wheel sqlite3 tesseract-ocr wget vim micro \
    curl ca-certificates imagemagick git ffmpeg && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm npx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash


EXPOSE 8000

# Add and install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Update PATH environment variable
ENV PATH="/root/.local/bin/:$PATH"
RUN apt update && apt install pup unzip \
    && rm -rf /var/lib/apt/lists/*


# Copy application files
COPY . /app/
RUN pip install -r requirements.txt
RUN uv sync --frozen

# Start the application
CMD ["uv", "run", "main_final.py"]
