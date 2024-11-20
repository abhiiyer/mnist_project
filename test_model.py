from model import EnhancedCNN
from train import train_model

def test_model():
    # Initialize and check parameter count
    model = EnhancedCNN()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 25000, f"Model has {param_count} parameters, exceeds limit!"
    print(f"Model parameter count test passed: {param_count} parameters")

    # Train model and check accuracy
    _, accuracy = train_model()
    assert accuracy >= 95, f"Training accuracy is {accuracy:.2f}%, less than 95%!"
    print("Model training accuracy test passed.")

if __name__ == "__main__":
    test_model()
