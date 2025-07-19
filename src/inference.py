import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

def main():
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Load trained model
    model = joblib.load("model_train.pkl")

    # Make predictions
    y_pred = model.predict(X)

    # Evaluate and print classification report
    print("Classification Report:\n")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()