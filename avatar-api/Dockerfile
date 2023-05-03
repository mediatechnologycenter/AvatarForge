FROM continuumio/miniconda3 as base
ENV DEBIAN_FRONTEND noninteractive

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

ENV TZ=Europe/Zurich

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# Update & install packages
RUN apt-get -qq update --fix-missing && \
    apt-get -qq install -y bash curl wget git ca-certificates openssh-client gpg tzdata && \
    apt-get -qq clean
    # Make sure google-cloud-sdk-gke-gcloud-auth-plugin is installed, because it won't be included in later releases

FROM base as backend
WORKDIR /app

COPY gunicorn.conf.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY avatar_backend_api ./avatar_backend_api

CMD [ "gunicorn", "-c", "gunicorn.conf.py", "--chdir", "./avatar_backend_api", "-k", "uvicorn.workers.UvicornWorker", "app:app" ]

