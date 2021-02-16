# mlflow_docker

Description: This repository contains an MLflow docker setup

## Containers
1. postgres database
    - Stores runs and metrics
    - Not accessible
2. minio s3 storage
    - Stores models and artifacts
    - Accessible via Port 9000
3. mlflow server
    - Orchestrates all tasks
    - Accessible via Port 5000 (via ngninx)
4. nginx webserver
    - Acts as authentication layer for the mlflow server

## Additional Notes
- Start with ``docker-compose -f "docker-compose.yml" up -d --build``
- Passwords and password-files in this git are only for testing purposes and will be changed in production
- Because of script locations and used proxies, the static variable REMOTE_IP and the env-variable PROXY could prevent successful execution