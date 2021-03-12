import os
import gzip
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from attribmod.utils import mltracker
from attribmod.data import abt_data
from attribmod.models.myclassifier import MyClassifier
from attribmod.utils import datafilefinder


def get_classifier(cusomized_clf: bool = False) -> MyClassifier:
    if cusomized_clf:
        filename = 'convmod_classifier_customized.pickle.gz'
    else:
        filename = 'convmod_classifier.pickle.gz'

    if 'MLFLOW_ARTIFACT_CONVMOD_CLASSIFIER' in os.environ:
        filepath = os.getenv('MLFLOW_ARTIFACT_CONVMOD_CLASSIFIER')
    else:
        filepath = datafilefinder.get_path('models', filename)
    with gzip.open(filepath, 'rb') as file:
        return pickle.load(file)


def fit_classifier(
    use_mlflow: bool = True, mlflow_comment: str = 'Ergebnis von abt_classifier', df=None
) -> None:
    if df is None:
        abt = abt_data.get_abt()
    else:
        abt = df.copy()

    cols = abt.columns.tolist()
    cols = sorted(cols)
    abt = abt[cols]

    xgbc = MyClassifier(sample_target_0=0.2)

    train, test = train_test_split(abt, test_size=0.2, random_state=42)
    X_train = train.drop(columns='antrag')
    y_train = train.antrag
    X_test = test.drop(columns='antrag')
    y_test = test.antrag

    if use_mlflow:
        with mltracker.start_run():
            xgbc.fit(X_train, y_train)
            y_pred = xgbc.predict(X_test)

            mltracker.log_metrics(y_test, y_pred)
            mltracker.mltracker().set_tag('Classifier', xgbc)
            mltracker.mltracker().set_tag('Features', X_train.columns.tolist())
            if mlflow_comment:
                mltracker.mltracker().set_tag('Comment', mlflow_comment)
    else:
        xgbc.fit(X_train, y_train)

    if df is None:
        filename = 'convmod_classifier.pickle.gz'
    else:
        filename = 'convmod_classifier_customized.pickle.gz'

    cls_path = Path(__file__).resolve().parent.parent.parent / 'models' / filename
    with gzip.open(cls_path, 'wb') as file:
        pickle.dump(xgbc, file)


def feature_importance() -> pd.DataFrame:
    clf = get_classifier()
    df = pd.DataFrame(
        dict(
            feature_name=clf.get_booster().feature_names,
            feature_importance=clf.feature_importances_,
        )
    )
    return df.sort_values('feature_importance', ascending=False)
