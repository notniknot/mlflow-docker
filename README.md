# mlflow_docker

This repository contains an MLflow docker setup with the following components:
- Docker Architecture to run MLflow with user authentication
- MLflow plugin that solves various problems
- Examples how to use MLflow

## Docker Environment
### Architecture
1. PostgreSQL database
    - Stores runs and metrics
    - Not accessible
2. Minio s3 storage
    - Stores models and artifacts
    - Accessible via Port 19000
3. MLflow server
    - Orchestrates all tasks
    - Accessible via Port 15000 (via ngninx)
4. nginx webserver
    - Acts as authentication layer for the mlflow server

![Docker Containers](docs/imgs/mlflow-docker-env.png)

### Usage
Make sure the directory structure is as follows:
```
docker
├── docker-compose.yml
├── .env
├── minio
│   └── data
│       └── mlflow-bucket
├── mlflow_server
│   ├── Dockerfile
│   └── mlflow_plugin-1.0.1-py3-none-any.whl
├── nginx
│   ├── Dockerfile
│   ├── .htpasswd
│   ├── mlflow.conf
│   └── nginx.conf
└── postgres
    └── data
```

- Rename `.env-template` file to `.env`
- Populate `.env` file with the proper values
- Decide whether you want to include the plugin or not
- Start with `docker-compose -f "docker-compose.yml" up -d --build`
## MLflow Plugin

### Features
- Configuration via YAML-File `mlflow-config.yaml`
- Set download path when downloading the model or artifacts
- Compare remote and local files before download:
	- Compare checksums
	- Compare version
- Use `.mlflowignore` to ignore specific files when uploading
- Just download certain files, such as a preprocessing pipeline
- Provide an interface to trigger actions on deployment
- Bug fix: Reduce S3-timeouts to speed up uploads
- Bug fix: Prevent S3-Content-Length-0-Bug when file is empty


### Installation
- Go to `plugin/` and build the wheel `python setup.py bdist_wheel`
- Install the plugin: `pip install mlflow_plugin-x.x.x-py3-none-any.whl`
- Fill `mlflow-config.yaml`

### Setting up the config file
Complete template with comments:
```yaml
# Section for MLflow-Server
MLFLOW:
  # Uri+Port for nginx
  MLFLOW_TRACKING_URI: http://127.0.0.1:15000
  # nginx basic http authentication
  MLFLOW_TRACKING_USERNAME: mlflow
  MLFLOW_TRACKING_PASSWORD: ***
# Section for S3-Minio-Storage
S3:
  # Uri+Port for Minio
  MLFLOW_S3_ENDPOINT_URL: http://127.0.0.1:19000
  # Minio user credentials
  AWS_ACCESS_KEY_ID: ***
  AWS_SECRET_ACCESS_KEY: ***
# Section for local settings
LOCAL_ENV:
  # Specified the environment the application is executed: dev, prod
  MLFLOW_ENV_TYPE: dev
  # 'true', 'yes' or '1' if empty files such as most __init__.py should be filled with '# empty file'
  # see boto3 bug: https://github.com/minio/minio/issues/6540 or https://github.com/urllib3/urllib3/issues/1438
  MLFLOW_FILL_EMPTY_UPLOAD_FILES: '1'
  # 'checksum' compares local and remote files by MD5 hash
  # 'version' compares local version with remote version
  MLFLOW_VALIDATE_FILES: checksum
  # Path where the model/artifacts should be downloaded
  MLFLOW_DST_PATH: /srv/dev/mlflow/examples/attribution_model/destination
  # Path to file with regex pattern for paths/files to be ignored
  MLFLOW_IGNORE_PATH: /srv/dev/mlflow/examples/attribution_model/.mlflowignore
  # Necessary if behind proxy
  no_proxy: localhost,127.0.0.1
```

## Examples
Examples that use various functions of the MLflow plugin
1. attribution_model
2. scikit-learn 'diabetes' toy dataset