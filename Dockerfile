FROM ubuntu:18.04



RUN apt-get update -y \
    && apt-get install -y python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
    && pip3 install flask flask_json transformers torch


ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

RUN mkdir -p bertinho

EXPOSE 8866

#Download nltk
WORKDIR /bertinho/
COPY ./ /bertinho/

CMD ["python3", "serve.py"]