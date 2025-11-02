import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from prepare_data import prepare_datasets
from model import FailurePredictor


def train_model(
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        df_train,
        df_test,
    ) = prepare_datasets()

    # numpy -> torch
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)  # shape (N,1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FailurePredictor(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(xb)

        avg_loss = epoch_loss / len(train_ds)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[epoch {epoch}] loss={avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(device)).cpu().numpy().flatten()

    # Binarize with threshold 0.5
    test_preds_binary = (test_preds >= 0.5).astype(int)
    y_true = y_test_t.numpy().flatten().astype(int)

    acc = accuracy_score(y_true, test_preds_binary)
    f1 = f1_score(y_true, test_preds_binary, zero_division=0)
    prec = precision_score(y_true, test_preds_binary, zero_division=0)
    rec = recall_score(y_true, test_preds_binary, zero_division=0)

    print("----- Evaluation -----")
    print("Accuracy :", acc)
    print("F1       :", f1)
    print("Precision:", prec)
    print("Recall   :", rec)

    # Save model and preprocessor
    torch.save(model.state_dict(), "model_checkpoint.pt")
    import joblib
    joblib.dump(preprocessor, "preprocessor.joblib")

    print("Artifacts saved: model_checkpoint.pt, preprocessor.joblib")


if __name__ == "__main__":
    train_model()
