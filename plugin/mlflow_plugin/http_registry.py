import os

from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_INSECURE_TLS_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
    _TRACKING_TOKEN_ENV_VAR,
    _TRACKING_USERNAME_ENV_VAR,
)
from mlflow.utils import rest_utils


class PluginHTTPRegistryStore(RestStore):
    @staticmethod
    def factory(store_uri, **_):
        # ToDo: Adapt to latest MLflow version
        def get_default_host_creds():
            return rest_utils.MlflowHostCreds(
                host=store_uri,
                username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
                password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
                token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
                ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == "true",
            )

        return PluginHTTPRegistryStore(get_default_host_creds)

    def __init__(self, get_host_creds):
        self.is_plugin = True
        super(PluginHTTPRegistryStore, self).__init__(get_host_creds)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        # print(
        #     f'CLI: Transitioned model {name} ({version}) to stage {stage}, archive_existing_versions: {archive_existing_versions}'
        # )
        return super(PluginHTTPRegistryStore, self).transition_model_version_stage(
            name, version, stage, archive_existing_versions
        )
