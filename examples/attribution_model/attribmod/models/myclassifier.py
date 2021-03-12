import numpy as np
from xgboost import XGBClassifier


class MyClassifier(XGBClassifier):

    def __init__(self, sample=None, sample_target_0=0.3, threshold=0.5, **kwargs):
        super().__init__(**kwargs)

        self.sample = sample
        self.sample_target_0 = sample_target_0
        self.threshold = threshold

    def fit(self, X, y, sample=None, sample_target_0=None, **kwargs):
        df = X.copy()
        if isinstance(y, np.ndarray):
            df['y'] = y
        else:
            df['y'] = y.values

        if sample is None:
            sample = self.sample
        if sample_target_0 is None:
            sample_target_0 = self.sample_target_0

        if sample_target_0 is not None and sample_target_0 < 1:
            df = df[df.y == 1].append(df[df.y == 0].sample(frac=sample_target_0))
        if sample is not None and sample < 1:
            df = df.sample(frac=sample)

        super().fit(df.drop(columns='y'), df.y, **kwargs)

    def predict(self, data, threshold=None, debug=False):
        if threshold is None:
            threshold = self.threshold if self.threshold else 0.5

        if debug:
            print(type(data))

        predict_proba = self.predict_proba(data)

        return (predict_proba[:, 1] >= threshold).astype(int)
