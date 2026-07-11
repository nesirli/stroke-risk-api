from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from stroke_risk.features.build_features import build_preprocessor


def build_log_reg_pipeline(C: float = 1.0, penalty: str = 'l2'):
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('model', LogisticRegression(
            C=C,
            penalty=penalty,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=1,
        ))
    ])