from src.data_loader import load_data
from src.preprocess import preprocess
from src.train import train_model

def run_training_pipeline():
    print("Loading data...")
    df = load_data("fetal_health.csv")

    print("Preprocessing data...")
    x_train, x_test, y_train, y_test = preprocess(df)

    print("Training model...")
    model = train_model(x_train, y_train)

    print("Model training completed and saved ✅")

if __name__ == "__main__":
    run_training_pipeline()