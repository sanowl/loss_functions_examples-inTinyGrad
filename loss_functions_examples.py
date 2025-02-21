import numpy as np
from tinygrad.tensor import Tensor
from contextlib import contextmanager
from sklearn.datasets import fetch_openml

@contextmanager
def no_grad():
    Tensor.no_grad = True
    try:
        yield
    finally:
        Tensor.no_grad = False
def fetch_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist['data'].astype(np.float32)
    Y = mnist['target'].astype(np.int32)
    X_train, X_test = X[:60000], X[60000:]
    Y_train, Y_test = Y[:60000], Y[60000:]
    return X_train, Y_train, X_test, Y_test    
def hinge_loss(y_true, y_pred):
    y_true_clipped = y_true.clip(-1, 1)
    loss = (1 - y_true_clipped * y_pred).relu()
    return loss.mean()

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred_clipped = y_pred.clip(epsilon, 1 - epsilon)
    loss = - (y_true * y_pred_clipped.log() + (1 - y_true) * (1 - y_pred_clipped).log())
    return loss.mean()
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred_clipped = y_pred.clip(epsilon, 1 - epsilon)
    loss = - (y_true * y_pred_clipped.log()).sum(axis=1).mean()
    return loss
def huber_loss(y_true, y_pred, delta=1.0):
    try:
        error = y_true - y_pred
        error_abs = error.abs()
        mask = (error_abs <= delta).astype(Tensor)
        small_error_loss = 0.5 * error ** 2
        big_error_loss = delta * (error_abs - 0.5 * delta)
        loss = small_error_loss * mask + big_error_loss * (1.0 - mask)
        return loss.mean()
    except Exception as e:
        raise ValueError(f"Error in huber_loss calculation: {e}")

def example_hinge_loss():
    print("Example 1: Hinge Loss for Binary Classification")
    data = Tensor(np.array([[0.5], [-1.2], [1.5], [-0.7]], dtype=np.float32))
    labels = Tensor(np.array([[1], [-1], [1], [-1]], dtype=np.float32))
    weights = Tensor.uniform(1, 1, requires_grad=True)
    learning_rate = 0.1
    for epoch in range(50):
        outputs = data * weights
        loss = hinge_loss(labels, outputs)
        loss.backward()
        with no_grad():
            weights_np = np.array(weights.data()).astype(np.float32)
            grad_np = np.array(weights.grad.data()).astype(np.float32)
            updated_weights = weights_np - grad_np * learning_rate
            weights.assign(updated_weights)
        weights.grad = None
        if epoch % 10 == 0:
            loss_value = np.array(loss.data()).astype(np.float32).item()
            print(f"Epoch {epoch}, Loss: {loss_value}")

def example_binary_cross_entropy_loss():
    print("\nExample 2: Binary Cross-Entropy Loss for Binary Classification")
    data = Tensor(np.array([[0.5], [-1.2], [1.5], [-0.7]], dtype=np.float32))
    labels = Tensor(np.array([[1], [0], [1], [0]], dtype=np.float32))
    weights = Tensor.uniform(1, 1, requires_grad=True)
    learning_rate = 0.1
    for epoch in range(50):
        logits = data * weights
        outputs = logits.sigmoid()
        loss = binary_cross_entropy_loss(labels, outputs)
        loss.backward()
        with no_grad():
            weights_np = np.array(weights.data()).astype(np.float32)
            grad_np = np.array(weights.grad.data()).astype(np.float32)
            updated_weights = weights_np - grad_np * learning_rate
            weights.assign(updated_weights)
        weights.grad = None
        if epoch % 10 == 0:
            loss_value = np.array(loss.data()).astype(np.float32).item()
            print(f"Epoch {epoch}, Loss: {loss_value}")

def example_cross_entropy_loss():
    print("\nExample 3: Cross-Entropy Loss for Multi-Class Classification")
    data = Tensor(np.array([[1.0, 2.0],
                            [1.5, -0.5],
                            [-1.0, 2.5],
                            [-1.5, -2.0]], dtype=np.float32))
    labels = Tensor(np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]], dtype=np.float32))
    weights = Tensor.uniform(2, 3, requires_grad=True)
    learning_rate = 0.1
    for epoch in range(100):
        logits = data.dot(weights)
        outputs = logits.softmax()
        loss = cross_entropy_loss(labels, outputs)
        loss.backward()
        with no_grad():
            weights_np = np.array(weights.data()).astype(np.float32)
            grad_np = np.array(weights.grad.data()).astype(np.float32)
            updated_weights = weights_np - grad_np * learning_rate
            weights.assign(updated_weights)
        weights.grad = None
        if epoch % 20 == 0:
            loss_value = np.array(loss.data()).astype(np.float32).item()
            print(f"Epoch {epoch}, Loss: {loss_value}")

def example_huber_loss():
    print("\nExample 4: Huber Loss for Regression with Anomalies")
    data = Tensor(np.linspace(-2, 2, num=100, dtype=np.float32).reshape(-1, 1))
    true_weights = 3.0
    noise = Tensor(np.random.normal(0, 0.5, size=(100, 1)).astype(np.float32))
    labels = data * true_weights + noise
    with no_grad():
        anomaly_noise = np.random.normal(0, 10, size=(10, 1)).astype(np.float32)
        labels_np = np.array(labels.data()).astype(np.float32)
        labels_np[:10] += anomaly_noise
        labels.assign(labels_np)
    weights = Tensor.uniform(1, 1, requires_grad=True)
    learning_rate = 0.05
    for epoch in range(200):
        predictions = data * weights
        loss = huber_loss(labels, predictions, delta=1.0)
        loss.backward()
        with no_grad():
            weights_np = np.array(weights.data()).astype(np.float32)
            grad_np = np.array(weights.grad.data()).astype(np.float32)
            updated_weights = weights_np - grad_np * learning_rate
            weights.assign(updated_weights)
        weights.grad = None
        if epoch % 40 == 0:
            loss_value = np.array(loss.data()).astype(np.float32).item()
            print(f"Epoch {epoch}, Loss: {loss_value}")
    estimated_weight = np.array(weights.data()).astype(np.float32).item()
    print(f"\nEstimated Weight: {estimated_weight}")

def example_mnist_cross_entropy():
    try:
        print("\nExample 5: Training a Neural Network on MNIST with Cross-Entropy Loss")
        X_train, Y_train, X_test, Y_test = fetch_mnist()
        
        # Ensure consistent data types
        X_train = (X_train.reshape(-1, 28*28) / 255.0).astype(np.float32)
        X_train_tensor = Tensor(X_train)
        
        Y_train_labels = Y_train.astype(np.int32)
        Y_train_one_hot = Tensor(np.eye(10, dtype=np.float32)[Y_train_labels])
        
        # Clear unused variables to free memory
        del X_train
        del Y_train
        
        weights_input_hidden = Tensor.uniform(28*28, 128, requires_grad=True)
        weights_hidden_output = Tensor.uniform(128, 10, requires_grad=True)
        learning_rate = 0.01
        epochs = 5
        batch_size = 64
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train_tensor.shape[0])
            # Ensure the shuffled data is float32
            X_train_shuffled = Tensor(np.array(X_train_tensor.data()).astype(np.float32)[permutation], requires_grad=False)
            Y_train_shuffled = Tensor(np.array(Y_train_one_hot.data()).astype(np.float32)[permutation], requires_grad=False)
            for i in range(0, X_train_tensor.shape[0], batch_size):
                x_batch = X_train_shuffled[i:i+batch_size]
                y_batch = Y_train_shuffled[i:i+batch_size]
                hidden_layer = x_batch.dot(weights_input_hidden).relu()
                outputs = hidden_layer.dot(weights_hidden_output).softmax()
                loss = cross_entropy_loss(y_batch, outputs)
                loss.backward()
                with no_grad():
                    weights_hidden_output_np = np.array(weights_hidden_output.data()).astype(np.float32)
                    weights_hidden_output_grad_np = np.array(weights_hidden_output.grad.data()).astype(np.float32)
                    updated_weights_hidden_output = weights_hidden_output_np - weights_hidden_output_grad_np * learning_rate
                    
                    weights_input_hidden_np = np.array(weights_input_hidden.data()).astype(np.float32)
                    weights_input_hidden_grad_np = np.array(weights_input_hidden.grad.data()).astype(np.float32)
                    updated_weights_input_hidden = weights_input_hidden_np - weights_input_hidden_grad_np * learning_rate
                    
                    weights_hidden_output.assign(updated_weights_hidden_output)
                    weights_input_hidden.assign(updated_weights_input_hidden)
                weights_hidden_output.grad = None
                weights_input_hidden.grad = None
            loss_value = np.array(loss.data()).astype(np.float32).item()
            print(f"Epoch {epoch+1}, Loss: {loss_value}")

        def calculate_accuracy(X, Y_labels):
            hidden_layer = X.dot(weights_input_hidden).relu()
            outputs = hidden_layer.dot(weights_hidden_output).softmax()
            predictions = np.array(outputs.data()).astype(np.float32).argmax(axis=1)
            accuracy = np.mean(predictions == Y_labels)
            return accuracy

        X_test_tensor = Tensor(X_test.reshape(-1, 28*28).astype(np.float32) / 255.0)
        test_accuracy = calculate_accuracy(X_test_tensor, Y_test.astype(np.int32))
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    except Exception as e:
        raise RuntimeError(f"MNIST training failed: {e}")
    finally:
        # Clean up tensors
        del X_train_tensor
        del Y_train_one_hot

if __name__ == "__main__":
    for example in [
        example_hinge_loss,
        example_binary_cross_entropy_loss,
        example_cross_entropy_loss,
        example_huber_loss,
        example_mnist_cross_entropy
    ]:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            continue
