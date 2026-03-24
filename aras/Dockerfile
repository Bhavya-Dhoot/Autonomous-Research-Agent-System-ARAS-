FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-publishers \
    texlive-fonts-recommended \
    ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY aras/requirements.txt /app/aras/requirements.txt
RUN pip install --no-cache-dir -r /app/aras/requirements.txt

COPY . /app

EXPOSE 8000

