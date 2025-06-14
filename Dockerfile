# Use Miniconda as base image
FROM continuumio/miniconda3

# Create a working directory
WORKDIR /app

# Copy requirements (if using pip-based install)
COPY requirements.txt .

# Create conda env, install Jupyter, and register kernel
RUN conda create -n ml-env python=3.9 -y && \
    conda run -n ml-env pip install --no-cache-dir -r requirements.txt && \
    conda run -n ml-env pip install ipykernel jupyter && \
    conda run -n ml-env python -m ipykernel install --user --name=ml-env --display-name "Python (ml-env)"

# Set environment variables
ENV CONDA_DEFAULT_ENV=ml-env
ENV PATH /opt/conda/envs/ml-env/bin:$PATH
