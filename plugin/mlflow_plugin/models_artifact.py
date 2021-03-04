import os

import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri


class PluginModelsArtifactRepository(ModelsArtifactRepository):
    """ModelsArtifactRepository provided through plugin system"""

    def __init__(self, artifact_uri):
        self.is_plugin = True
        return super(PluginModelsArtifactRepository, self).__init__(artifact_uri)

    def _get_remote_version(self, uri):
        from mlflow.tracking import MlflowClient

        databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
        )
        client = MlflowClient(registry_uri=databricks_profile_uri)
        _, version = get_model_name_and_version(client, self.artifact_uri)
        return version

    def _set_new_local_version(self, version_file, new_version):
        with open(version_file, 'w') as file:
            file.write(new_version)

    def _get_local_and_remote_versions(self, dst_path, artifact_path):
        if dst_path is None:
            return False

        remote_version = self._get_remote_version(artifact_path)

        version_file = os.path.join(dst_path, artifact_path, '.mlflowversion')
        version_dirname = os.path.join(dst_path, artifact_path)
        if not os.path.exists(version_dirname):
            os.makedirs(version_dirname, exist_ok=True)
        if not os.path.exists(version_file):
            self._set_new_local_version(version_file, remote_version)
            return {
                'local_version': None,
                'remote_version': remote_version,
                'version_file': version_file,
            }

        with open(version_file, 'r') as file:
            local_version = file.read().strip()

        return {
            'local_version': local_version,
            'remote_version': remote_version,
            'version_file': version_file,
        }

    def download_artifacts(self, artifact_path, dst_path=None):
        if dst_path is None:
            dst_path = os.environ.get('MLFLOW_DST_PATH', dst_path)
        if os.environ.get('MLFLOW_VALIDATE_FILES', '').lower() == 'version':
            versions = self._get_local_and_remote_versions(dst_path, artifact_path)
            if versions['remote_version'] == versions['local_version']:
                print(
                    f'Local version ({versions["local_version"]}) already up to date. Skipping download.'
                )
                return os.path.join(dst_path, artifact_path)

        full_artifact_path = super(PluginModelsArtifactRepository, self).download_artifacts(
            artifact_path, dst_path
        )
        if 'versions' in locals():
            self._set_new_local_version(versions['version_file'], versions['remote_version'])
            print(f'Setting new local version to {versions["remote_version"]}.')
        return full_artifact_path
