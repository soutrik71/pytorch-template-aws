## __POETRY SETUP__

```bash
# Install poetry
conda create -n poetry_env python=3.10 -y
conda activate poetry_env
pip install poetry
poetry env info
poetry new pytorch_project
cd pytorch_project/
# fill up the pyproject.toml file without pytorch and torchvision
poetry install

# Add dependencies to the project for pytorch and torchvision
poetry source add --priority explicit pytorch_cpu https://download.pytorch.org/whl/cpu
poetry add --source pytorch_cpu torch torchvision
poetry lock
poetry show
poetry install --no-root

# Add dependencies to the project 
poetry add matplotlib
poetry add hydra-core
poetry add omegaconf
poetry add hydra_colorlog
poetry add --dev black # 
poetry lock
poetry show

Type	Purpose	Installation Command
  Normal Dependency	Required for the app to run in production.	poetry add <package>
  Development Dependency	Needed only during development (e.g., testing, linting).	poetry add --dev <package>
# Add dependencies to the project with specific version
poetry add <package_name>@<version>
```

## __MULTISTAGEDOCKER SETUP__

#### Step-by-Step Guide to Creating Dockerfile and docker-compose.yml for a New Code Repo

If you're new to the project and need to set up Docker and Docker Compose to run the training and inference steps, follow these steps.

---

### 1. Setting Up the Dockerfile

A Dockerfile is a set of instructions that Docker uses to create an image. In this case, we'll use a __multi-stage build__ to make the final image lightweight while managing dependencies with `Poetry`.

#### Step-by-Step Process for Creating the Dockerfile

1. __Choose a Base Image__:
   - We need to choose a Python image that matches the project's required version (e.g., Python 3.10.14).
   - Use the lightweight __`slim`__ version to minimize image size.

   ```Dockerfile
   FROM python:3.10.14-slim as builder
   ```

2. __Install Dependencies in the Build Stage__:
   - We'll use __Poetry__ for dependency management. Install it using `pip`.
   - Next, copy the `pyproject.toml` and `poetry.lock` files to the `/app` directory to install dependencies.

   ```Dockerfile
   RUN pip3 install poetry==1.7.1
   WORKDIR /app
   COPY pytorch_project/pyproject.toml pytorch_project/poetry.lock /app/
   ```

3. __Configure Poetry__:
   - Configure Poetry to install the dependencies in a virtual environment inside the project directory (not globally). This keeps everything contained and avoids conflicts with the system environment.

   ```Dockerfile
   ENV POETRY_NO_INTERACTION=1 \
       POETRY_VIRTUALENVS_IN_PROJECT=1 \
       POETRY_VIRTUALENVS_CREATE=true \
       POETRY_CACHE_DIR=/tmp/poetry_cache
   ```

4. __Install Dependencies__:
   - Use `poetry install --no-root` to install only the dependencies and not the package itself. This is because you typically don't need to install the actual project code at this stage.

   ```Dockerfile
   RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root
   ```

5. __Build the Runtime Stage__:
   - Now, set up the final runtime image. This stage will only include the required application code and the virtual environment created in the first stage.
   - The final image will use the same Python base image but remain small by avoiding the re-installation of dependencies.

   ```Dockerfile
   FROM python:3.10.14-slim as runner
   WORKDIR /app
   COPY src /app/src
   COPY --from=builder /app/.venv /app/.venv
   ```

6. __Set Up the Path to Use the Virtual Environment__:
   - Update the `PATH` environment variable to use the Python binaries from the virtual environment.

   ```Dockerfile
   ENV PATH="/app/.venv/bin:$PATH"
   ```

7. __Set a Default Command__:
   - Finally, set the command that will be executed by default when the container is run. You can change or override this later in the Docker Compose file.

   ```Dockerfile
   CMD ["python", "-m", "src.train"]
   ```

### Final Dockerfile

```Dockerfile
# Stage 1: Build environment with Poetry and dependencies
FROM python:3.10.14-slim as builder
RUN pip3 install poetry==1.7.1
WORKDIR /app
COPY pytorch_project/pyproject.toml pytorch_project/poetry.lock /app/
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Stage 2: Runtime environment
FROM python:3.10.14-slim as runner
WORKDIR /app
COPY src /app/src
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "src.train"]
```

---

### 2. Setting Up the docker-compose.yml File

The `docker-compose.yml` file is used to define and run multiple Docker containers as services. In this case, we need two services: one for __training__ and one for __inference__.

### Step-by-Step Process for Creating docker-compose.yml

1. __Define the Version__:
   - Docker Compose uses a versioning system. Use version `3.8`, which is widely supported and offers features such as networking and volume support.

   ```yaml
   version: '3.8'
   ```

2. __Set Up the `train` Service__:
   - The `train` service is responsible for running the training script. It builds the Docker image, runs the training command, and uses volumes to store the data, checkpoints, and artifacts.

   ```yaml
   services:
     train:
       build:
         context: .
       command: python -m src.train
       volumes:
         - data:/app/data
         - checkpoints:/app/checkpoints
         - artifacts:/app/artifacts
       shm_size: '2g'  # Increase shared memory to prevent DataLoader issues
       networks:
         - default
       env_file:
         - .env  # Load environment variables
   ```

3. __Set Up the `inference` Service__:
   - The `inference` service runs after the training has completed. It waits for a file (e.g., `train_done.flag`) to be created by the training process and then runs the inference script.

   ```yaml
     inference:
       build:
         context: .
       command: /bin/bash -c "while [ ! -f /app/checkpoints/train_done.flag ]; do sleep 10; done; python -m src.infer"
       volumes:
         - checkpoints:/app/checkpoints
         - artifacts:/app/artifacts
       shm_size: '2g'
       networks:
         - default
       depends_on:
         - train
       env_file:
         - .env
   ```

4. __Define Shared Volumes__:
   - Volumes allow services to share data. Here, we define three shared volumes:
     - `data`: Stores the input data.
     - `checkpoints`: Stores the model checkpoints and the flag indicating training is complete.
     - `artifacts`: Stores the final model outputs or artifacts.

   ```yaml
   volumes:
     data:
     checkpoints:
     artifacts:
   ```

5. __Set Up Networking__:
   - Use the default network to allow the services to communicate.

   ```yaml
   networks:
     default:
   ```

### Final docker-compose.yml

```yaml
version: '3.8'

services:
  train:
    build:
      context: .
    command: python -m src.train
    volumes:
      - data:/app/data
      - checkpoints:/app/checkpoints
      - artifacts:/app/artifacts
    shm_size: '2g'
    networks:
      - default
    env_file:
      - .env

  inference:
    build:
      context: .
    command: /bin/bash -c "while [ ! -f /app/checkpoints/train_done.flag ]; do sleep 10; done; python -m src.infer"
    volumes:
      - checkpoints:/app/checkpoints
      - artifacts:/app/artifacts
    shm_size: '2g'
    networks:
      - default
    depends_on:
      - train
    env_file:
      - .env

volumes:
  data:
  checkpoints:
  artifacts:

networks:
  default:
```

---

### Summary

1. __Dockerfile__:
   - A multi-stage Dockerfile is used to create a lightweight image where the dependencies are installed with Poetry and the application code is run using a virtual environment.
   - It ensures that all dependencies are isolated in a virtual environment, and the final container only includes what is necessary for the runtime.

2. __docker-compose.yml__:
   - The `docker-compose.yml` file defines two services:
     - __train__: Runs the training script and stores checkpoints.
     - __inference__: Waits for the training to finish and runs inference based on the saved model.
   - Shared volumes ensure that the services can access data, checkpoints, and artifacts.
   - `shm_size` is increased to prevent issues with DataLoader in PyTorch when using multiple workers.

This setup allows for easy management of multiple services using Docker Compose, ensuring reproducibility and simplicity.

## __References__

- <https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker>
- <https://github.com/fralik/poetry-with-private-repos/blob/master/Dockerfile>
- <https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0>
- <https://www.martinrichards.me/post/python_poetry_docker/>
- <https://gist.github.com/soof-golan/6ebb97a792ccd87816c0bda1e6e8b8c2>

8. ## __DVC SETUP__

First, install dvc using the following command

```bash
dvc init
dvc version
dvc init -f
dvc config core.autostage true
dvc add data
dvc remote add -d myremote /tmp/dvcstore
dvc push
```

Add some more file in the data directory and run the following commands

```bash
dvc add data
dvc push
dvc pull
```

Next go back to 1 commit and run the following command

```bash
git checkout HEAD~1
dvc checkout
# you will get one file less
```

Next go back to the latest commit and run the following command

```bash
git checkout -
dvc checkout
dv pull
dvc commit
```

Next run the following command to add google drive as a remote

```bash
dvc remote add --default gdrive gdrive://1w2e3r4t5y6u7i8o9p0
dvc remote modify gdrive gdrive_acknowledge_abuse true
dvc remote modify gdrive gdrive_client_id <>
dvc remote modify gdrive gdrive_client_secret <>
# does not work when used from VM and port forwarding to local machine
```

Next run the following command to add azure-blob as a remote

```bash
dvc remote remove azblob
dvc remote add --default azblob azure://mycontainer/myfolder
dvc remote modify --local azblob connection_string "<>"
dvc remote modify azblob  allow_anonymous_login true
dvc push -r azblob
# this works when used and requires no explicit login
```

Next we will add S3 as a remote

```bash
dvc remote add --default aws_remote s3://deep-bucket-s3/data
dvc remote modify --local aws_remote access_key_id <>
dvc remote modify --local aws_remote secret_access_key <>
dvc remote modify --local aws_remote region ap-south-1
dvc remote modify aws_remote region ap-south-1
dvc push -r aws_remote -v
```

9. ## __HYDRA SETUP__

```bash
# Install hydra
pip install hydra-core hydra_colorlog omegaconf
# Fillup the configs folder with the files as per the project
# Run the following command to run the hydra experiment
# for train 
python -m src.hydra_test experiment=catdog_experiment ++task_name=train ++train=True ++test=False
# for eval
python -m src.hydra_test experiment=catdog_experiment ++task_name=eval ++train=False ++test=True
# for both
python -m src.hydra_test experiment=catdog_experiment task_name=train train=True test=True # + means adding new key value pair to the existing config and ++ means overriding the existing key value pair
```

10. ## __LOCAL SETUP__

```bash
 python -m src.train experiment=catdog_experiment ++task_name=train ++train=True ++test=False
 python -m src.train experiment=catdog_experiment ++task_name=eval ++train=False ++test=True
 python -m src.infer experiment=catdog_experiment
```

11. ## _DVC_PIPELINE_SETUP_

```bash
dvc repro
```
12. ## _DVC Experiments_
  - To run the dvc experiments keep different experiment_<>.yaml files in the configs folder under experiment folder
  - Make sure to override the default values in the experiment_<>.yaml file for each parameter that you want to change

13. ## _HYDRA Experiments_
  - make sure to declare te config file in yaml format in the configs folder hparam
  - have hparam null in train and eval config file
  - run the following command to run the hydra experiment
  ```bash
   python -m src.train --multirun experiment=catdog_experiment_convnext ++task_name=train ++train=True ++test=False hparam=catdog_classifier_covnext
   python -m src.create_artifacts
  ```

14. ## __Latest Execution Command__

```bash
python -m src.train_optuna_callbacks experiment=catdog_experiment ++task_name=train ++train=True ++test=False
python -m src.train_optuna_callbacks experiment=catdog_experiment ++task_name=test ++train=False ++test=True
python -m src.infer experiment=catdog_experiment
```

15. ## __GPU Setup__
```bash
docker build -t my-gpu-app .
docker run --gpus all my-gpu-app
docker exec -it <container_id> /bin/bash
# pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime supports cuda 12.1 and python 3.10.14
```
```bash
# for docker compose what we need to is follow similar to the following
services:
  test:
    image: nvidia/cuda:12.3.1-base-ubuntu20.04
    command: nvidia-smi
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```