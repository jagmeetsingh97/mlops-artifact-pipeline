import json
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

def main():
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    import os

    config_path = os.path.join(os.path.dirname(__file__), "..", "Config", "Config.json")
    config_path = os.path.abspath(config_path)  # Converts to full path
    #config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
    config = load_config(config_path)
    # Train model
    model = train_model(X, y, config)

    # Save model
    joblib.dump(model, "model_train.pkl")
    print("Model trained and saved as model_train.pkl")

if __name__ == "__main__":
    main()