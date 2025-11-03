# Use Miniconda as base image
# FROM continuumio/miniconda3


# # Install system-level dependencies needed for OpenCV
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev\
#     && rm -rf /var/lib/apt/lists/* 



# # Create a working directory
# WORKDIR /app

# # Copy requirements (if using pip-based install)
# COPY requirements.txt .




# # Create conda env, install Jupyter, and register kernel
# RUN conda create -n ml-env python=3.9 -y && \
#     conda run -n ml-env pip install --no-cache-dir -r requirements.txt && \
#     conda run -n ml-env pip install ipykernel jupyter && \
#     conda run -n ml-env python -m ipykernel install --user --name=ml-env --display-name "Python (ml-env)"

# # Set environment variables
# ENV CONDA_DEFAULT_ENV=ml-env
# ENV PATH /opt/conda/envs/ml-env/bin:$PATH



# Use Miniconda as base image
FROM continuumio/miniconda3

# System libs for OpenCV, PyAV/WebRTC, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    # (optional) only if you plan to access /dev/video* directly inside container
    libsm6 libxext6 libxrender1 libgomp1 libgtk-3-0 \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# App root
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Create env and install deps (headless OpenCV recommended in containers)
RUN conda create -n ml-env python=3.9 -y && \
    conda run -n ml-env python -m pip install --no-cache-dir --upgrade pip && \
    conda run -n ml-env python -m pip install --no-cache-dir -r /app/requirements.txt

# Ensure shells and PATH use the env
SHELL ["bash", "-lc"]
RUN echo "conda activate ml-env" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=ml-env
ENV PATH=/opt/conda/envs/ml-env/bin:/opt/conda/bin:$PATH

# Copy the rest of your app
COPY . /app

# Output dir (if you save images)
RUN mkdir -p /app/outputs

# Streamlit env
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXsrfProtection=false \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Health check (needs curl installed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Always run inside the conda env
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml-env"]
CMD ["streamlit", "run", "src/main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

