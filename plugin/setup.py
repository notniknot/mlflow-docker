from setuptools import setup, find_packages

# python setup.py bdist_wheel

# ToDo: If not downloadable use local modal (error handling)

# Done
# - Konfiguration über YAML-File
# - Pfadangabe bei Model-Download
# - Abgleich der Files bei Download
# 	-> Vergleichen der Checksums
# 	-> Vergleichen der Versionen
# - .mlflowignore bei File-Upload
# - Download nur der Preprocessing-Komponente
# - Schnittstelle für Deployment bereitgestellt
# - Timeout reduzieren, damit log schneller geht
# - Boto3 Content-Length-0-Fix


setup(
    name="mlflow_plugin",
    version="1.0.1",
    description="Cosmos MLflow Plugin",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={
        "mlflow.artifact_repository": [
            # s3://... uris
            "s3=mlflow_plugin.s3_artifact:PluginS3ArtifactRepository",
            # models:/... uris
            "models=mlflow_plugin.models_artifact:PluginModelsArtifactRepository",
        ],
        "mlflow.model_registry_store": [
            # http://... uris -> via cli
            "http=mlflow_plugin.http_registry:PluginHTTPRegistryStore.factory",
            # postgresql://... uris -> via website
            "postgresql=mlflow_plugin.postgresql_registry:PluginRegistryPostgreSQLStore",
        ],
    },
)
