
# Use the official image as a parent image.
FROM ubuntu:22.04

# Update the system
RUN apt-get -y update && apt-get install -y bash \
    build-essential \
    wget \
    make \
    git \
    curl \
    sudo \
    ca-certificates \
    poppler-utils \
    python3 \
    python3-pip \
    nodejs \
    npm && \
    rm -rf /var/lib/apt/lists/*

# Install python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    pandas \
    openpyxl \
    matplotlib \
    numpy \
    wordcloud \
    scikit-learn \
    seaborn \
    ipywidgets \
    widgetsnbextension \
    jupyter_contrib_nbextensions \
    jupyter_nbextensions_configurator \
    jupyterlab-git \
    gradio \
    streamlit \
    streamlit_option_menu \
    nb-clean \
    plotly \
    openai \
    nbdime \
    transformers \
    torch \
    base64 \
    io \
    PIL \
    nbdev

# Copy requirements.txt and install Python libraries
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Run nbdev_install_quarto
RUN sudo nbdev_install_quarto

# Indicate the Dockerfile owner
LABEL maintainer="Team-Hummingbird"

CMD [ "/bin/bash" ]
