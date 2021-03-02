from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore


class PluginRegistryPostgreSQLStore(SqlAlchemyStore):
    def __init__(self, store_uri=None):
        self.is_plugin = True
        super(PluginRegistryPostgreSQLStore, self).__init__(store_uri)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        print(
            f'WEB: Transitioned model {name} ({version}) to stage {stage}, archive_existing_versions: {archive_existing_versions}'
        )
        return super(PluginRegistryPostgreSQLStore, self).transition_model_version_stage(
            name, version, stage, archive_existing_versions
        )
