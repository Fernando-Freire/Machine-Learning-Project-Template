################################################################################
##                                BASE IMAGE                                  ##
################################################################################

# Use Python 3.10-slim
FROM python:3.10.4 AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_VERSION=1.1.13 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/usr/src/poetry_cache/

################################################################################
##                              JUPYTER IMAGE                                 ##
################################################################################

# Build image for Jupyter
FROM base AS development

# Environment variables to configure JupyterLab
ENV JUPYTER_HOST="0.0.0.0" \
    JUPYTER_PORT="8888" \
    JUPYTER_GROUP="jovyan" \
    JUPYTER_USER="jovyan" \
    JUPYTER_HOME_DIR="/home/jovyan" \
    JUPYTER_NB_DIR="/usr/src/notebooks"

# Work directory inside container
WORKDIR ${JUPYTER_NB_DIR}

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install Packages
COPY pyproject.toml poetry.lock ./
RUN poetry update
RUN poetry install  --no-interaction --no-ansi --no-root --no-dev 

# Install Jupyter packages
ADD requirements_jupyter.txt ./
RUN pip install -r requirements_jupyter.txt

# Add all notebooks to the container
ADD ./notebooks ./

#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#RUN unzip awscliv2.zip
#RUN ./aws/install

# Build arguments to set non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create non-root user jovyan and give appropriate permissions
RUN groupadd -g ${GROUP_ID} ${JUPYTER_GROUP} \
  && useradd -u ${USER_ID} -g ${JUPYTER_GROUP} ${JUPYTER_USER} \
  && mkdir ${JUPYTER_HOME_DIR} \
  && chown ${USER_ID}:${GROUP_ID} ${JUPYTER_HOME_DIR} ${JUPYTER_NB_DIR}

# Set non-root user jovyan to run JupyterLab
USER ${USER_ID}:${GROUP_ID}

# Expose JupyterLab to outside the container
EXPOSE ${JUPYTER_PORT}

# By default, execute JupyterLab
CMD jupyter notebook --ip ${JUPYTER_HOST} --port ${JUPYTER_PORT} --no-browser --notebook-dir ${JUPYTER_NB_DIR}

################################################################################
##                              TESTING IMAGE                                 ##
################################################################################

#Build image for tests execution
FROM base as test

ENV SCRIPTS_GROUP="scripts" \
    SCRIPTS_USER="scripts" \
    SCRIPTS_HOME_DIR="/home/scripts" \
    SCRIPTS_DIR="/usr/src/scripts"

WORKDIR ${SCRIPTS_DIR}

#Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install Packages
ADD pyproject.toml poetry.lock ./
RUN poetry install  --no-interaction --no-ansi --no-root

# Add all scripts files to the container
ADD ./scripts .

# Build arguments to set non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create non-root user scripts and give appropriate permissions
RUN groupadd -g ${GROUP_ID} ${SCRIPTS_GROUP} \
  && useradd -u ${USER_ID} -g ${SCRIPTS_GROUP} ${SCRIPTS_USER} \
  && mkdir ${SCRIPTS_HOME_DIR} \
  && chown ${USER_ID}:${GROUP_ID} ${SCRIPTS_HOME_DIR} ${SCRIPTS_NB_DIR}

# Set non-root user scritps to run JupyterLab
USER ${USER_ID}:${GROUP_ID}

ENTRYPOINT ["python"]
