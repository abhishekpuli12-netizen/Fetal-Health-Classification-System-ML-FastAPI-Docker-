from sklearn.model_selection import train_test_split

def preprocess(df):
    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"] - 1  # convert labels to 0,1,2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test