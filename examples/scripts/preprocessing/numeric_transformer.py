from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Test
def get_numeric_transformer():
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
    )
    return numeric_transformer
