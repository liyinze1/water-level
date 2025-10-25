"""
Lottery Ticket Hypothesis (Frankle & Carbin, 2019) proposes that:

A randomly-initialized dense neural network contains a subnetwork (a "winning ticket") that, when trained in isolation, can match the performance of the full model.

Ref:
The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.” 
arXiv preprint arXiv:1803.03635 (2019).
https://arxiv.org/abs/1803.03635

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -------------------------
# Training and evaluation
# -------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

def test(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)

# -------------------------
# Pruning Function
# -------------------------
def prune_by_magnitude(model, prune_percent):
    # Flatten all weights
    all_weights = torch.cat([param.view(-1).abs() for name, param in model.named_parameters() if "weight" in name])
    threshold = torch.quantile(all_weights, prune_percent)

    mask_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = (param.abs() > threshold).float()
            mask_dict[name] = mask
    return mask_dict

# -------------------------
# Apply Mask
# -------------------------
def apply_mask(model, mask_dict):
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data *= mask_dict[name]

# -------------------------
# Main LTH workflow
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=64, shuffle=False)

# Initial training
model = SimpleNet().to(device)
initial_state = {k: v.clone() for k, v in model.state_dict().items()}  # save initial weights

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 3
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, criterion, device)
acc_before = test(model, test_loader, device)
print(f"Accuracy before pruning: {acc_before:.4f}")

# Prune 80% of weights
mask_dict = prune_by_magnitude(model, prune_percent=0.8)
apply_mask(model, mask_dict)

# Reset weights to original initialization
model.load_state_dict(initial_state)
apply_mask(model, mask_dict)  # reapply pruning mask

# Retrain the sparse subnetwork
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, criterion, device)
acc_after = test(model, test_loader, device)
print(f"Accuracy after retraining pruned model: {acc_after:.4f}")
