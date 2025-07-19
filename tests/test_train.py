import os
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits



import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import load_config, train_model
@pytest.fixture
def digits_data():
    digits = load_digits()
    return digits.data, digits.target

def test_config_loading():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
    config = load_config(config_path)
    assert "C" in config
    assert "solver" in config
    assert "max_iter" in config
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_type(digits_data):
    X, y = digits_data
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
    config = load_config(config_path)
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)

def test_model_accuracy(digits_data):
    X, y = digits_data
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
    config = load_config(config_path)
    model = train_model(X, y, config)
    score = model.score(X, y)
    assert score > 0.8  # sanity check threshold