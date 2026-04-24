from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier 
from sklearn.utils.class_weight import compute_sample_weight 
import joblib
import os

def train_model(x_train, y_train):

    weights = compute_sample_weight('balanced', y_train)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='mlogloss'
        ))
    ])

    pipe.fit(x_train, y_train, model__sample_weight=weights)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/model.pkl")

    print("Model saved successfully ✅")

    return pipe