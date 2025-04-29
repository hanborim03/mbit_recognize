import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

def train(model, X_train, y_train, X_test, y_test, class_weights, epochs=5, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # 평가
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        predicted = torch.argmax(preds, dim=1).cpu().numpy()
        accuracy = (predicted == y_test_tensor.cpu().numpy()).mean()
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test_tensor.cpu().numpy(), predicted))
