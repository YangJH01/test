import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 함수 정의
def train_model(X_train, y_train, X_val, y_val, activation_fn, lr= 0.001, epochs=20):
    model = nn.Sequential(
        nn.Linear(7,16),
        nn.ReLU(),
        nn.Linear(16,2)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())