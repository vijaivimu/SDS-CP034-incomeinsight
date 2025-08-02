
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer, 
    OrdinalEncoder, 
    PolynomialFeatures
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin 

# ----- 1. Custom transformer for binning education_num -----
class EducationBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be a 2D array or DataFrame, get the first column
        X = pd.Series(X.ravel())
        return X.apply(self._bin_education).to_frame()
    
    def _bin_education(self, edu_num):
        if edu_num <= 4:
            return 'PreHighSchool'
        elif edu_num <= 7:
            return 'IncompleteHS'
        elif edu_num <= 10:
            return 'HighSchool'
        elif edu_num <= 12:
            return 'Associate'
        elif edu_num == 13:
            return 'Bachelors'
        else:
            return 'Advanced'
        
# ----- 2. Apply column-level binary transformations -----
def map_binary_cols(df):
    df = df.copy()
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})
    df['married_together'] = df['marital.status'].map({
        'Married-AF-spouse': 1,
        'Married-civ-spouse': 1
    }).fillna(0)
    df['has_capital_gain'] = (df['capital.gain'] > 0).astype(int)
    return df

# Preprocessor Builder
# ------------------------
def build_preprocessor():
    # Define column groups
    cat_interact = ['marital.status', 'occupation', 'workclass']
    numeric_interact = ['hours.per.week', 'education.num', 'age', 'capital.gain']
    onehot_cols = ['native.country', 'race']
    edu_num_col = ['education.num']
    normal_cols = ['age', 'hours.per.week', 'education.num']
    skewed_col = ['capital.gain']
    bin_flags = ['income', 'sex', 'marital.status']

    # Pipelines
    cat_interact_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first')),
        ('interact', PolynomialFeatures(interaction_only=True, include_bias=False))
    ])

    numeric_interaction_pipeline = Pipeline([
        ('interact', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    edu_pipeline = Pipeline([
        ('bin', EducationBinner()),
        ('ord', OrdinalEncoder(categories=[[
            'Low', 'Incomplete', 'HighSchool', 
            'Associate', 'Bachelors', 'Advanced'
        ]]))
    ])

    onehot_pipeline = OneHotEncoder(drop='first')

    normal_pipeline = StandardScaler()
    skewed_pipeline = MinMaxScaler()

    # ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('binary_flags', FunctionTransformer(map_binary_cols, validate=False), bin_flags),
        ('interact', cat_interact_pipeline, cat_interact),
        ('edu_bin_ord', edu_pipeline, edu_num_col),
        ('onehot', onehot_pipeline, onehot_cols),
        ('normal', normal_pipeline, normal_cols),
        ('skewed', skewed_pipeline, skewed_col),
        ('interact_num', numeric_interaction_pipeline, numeric_interact)
    ], remainder='passthrough')  # To keep binary columns like sex, has_capital_gain, etc.

    return preprocessor

# Final pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit and transform
data_transformed = pipeline.fit_transform(data_cleaned)

dump(pipeline, '../models/model_pipeline.pkl')