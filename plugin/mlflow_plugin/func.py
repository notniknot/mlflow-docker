import os
import posixpath
import sys
import urllib.parse
from pathlib import Path

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository


def get_preprocessing(
    model_uri: str,
    preprocessing_location: str = 'preprocessing',
    artifacts: list = None,
    dst_path: str = None,
    append_to_sys_path: bool = True,
) -> dict:
    parsed_uri = urllib.parse.urlparse(str(model_uri))
    prefix = ""
    if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
        # relative path is a special case, urllib does not reconstruct it properly
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    # For models:/ URIs, it doesn't make sense to initialize a ModelsArtifactRepository with only
    # the model name portion of the URI, then call download_artifacts with the version info.
    if ModelsArtifactRepository.is_models_uri(model_uri):
        root_uri = model_uri
        rel_path = ""
    else:
        rel_path = posixpath.basename(parsed_uri.path) + '/'
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)
    rel_path = rel_path.lstrip('/\\')

    if dst_path is None:
        dst_path = os.environ.get('MLFLOW_DST_PATH', dst_path)

    artifact_repo = get_artifact_repository(artifact_uri=root_uri)
    paths = {'preprocessing': '', 'artifacts': {}}

    preprocessing_path = rel_path + f'code/{preprocessing_location}'.rstrip('/\\')
    artifact_location = artifact_repo.download_artifacts(
        artifact_path=preprocessing_path, dst_path=dst_path
    )
    paths['preprocessing'] = artifact_location

    if isinstance(artifacts, list):
        for artifact in artifacts:
            artifact_path = rel_path + f'artifacts/{artifact}'.rstrip('/\\')
            artifact_location = artifact_repo.download_artifacts(
                artifact_path=artifact_path, dst_path=dst_path
            )
            paths['artifacts'][artifact] = artifact_location

    if append_to_sys_path:
        sys.path.append(str(Path(paths['preprocessing']).resolve().parent))

    return paths
