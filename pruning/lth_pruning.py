"""
This is an iterative pruning implementation of the Lottery Ticket Hypothesis (LTH) in PyTorch
based on Frankle & Carbin’s 2019 methodology.


Ref.
Frankle, Jonathan, and Michael Carbin.
“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.” arXiv preprint arXiv:1803.03635 (2019).
https://arxiv.org/abs/1803.03635

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# -------------------------
# Model definition
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Training and testing
# -------------------------
def train(model, loader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
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
# Pruning utilities
# -------------------------
def prune_by_magnitude(model, mask_dict, prune_fraction):
    # Collect all unmasked weights
    all_weights = torch.cat([
        (param[mask_dict[name] == 1]).abs().view(-1)
        for name, param in model.named_parameters()
        if "weight" in name
    ])
    threshold = torch.quantile(all_weights, prune_fraction)

    new_mask_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            # keep weights above threshold, preserve existing mask
            mask = (param.abs() > threshold).float() * mask_dict[name]
            new_mask_dict[name] = mask
        else:
            new_mask_dict[name] = torch.ones_like(param)
    return new_mask_dict

def apply_mask(model, mask_dict):
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data *= mask_dict[name]

# -------------------------
# Main LTH iterative loop
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform),
                          batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform),
                         batch_size=128, shuffle=False)

model = SimpleNet().to(device)
initial_state = copy.deepcopy(model.state_dict())  # Save initial random initialization
criterion = nn.CrossEntropyLoss()

# Initialize mask (all ones)
mask_dict = {name: torch.ones_like(param) for name, param in model.named_parameters()}

num_rounds = 5        # number of iterative pruning rounds
prune_fraction = 0.2  # prune 20% each round
epochs_per_round = 3

for round_idx in range(num_rounds):
    print(f"\n=== Iteration {round_idx+1}/{num_rounds} ===")

    # 1. Reset to initial weights and reapply mask
    model.load_state_dict(copy.deepcopy(initial_state))
    apply_mask(model, mask_dict)

    # 2. Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, optimizer, criterion, device, epochs=epochs_per_round)
    acc = test(model, test_loader, device)
    print(f"Accuracy before pruning: {acc:.4f}")

    # 3. Prune weights
    mask_dict = prune_by_magnitude(model, mask_dict, prune_fraction)

    # 4. Apply mask immediately to zero pruned weights
    apply_mask(model, mask_dict)

    # Check remaining sparsity
    total_params = 0
    remaining_params = 0
    for name, mask in mask_dict.items():
        total_params += mask.numel()
        remaining_params += mask.sum().item()
    sparsity = 100 * (1 - remaining_params / total_params)
    print(f"Sparsity after pruning: {sparsity:.2f}%")

# Final retraining on the last winning ticket
print("\n=== Retraining final winning ticket ===")
model.load_state_dict(copy.deepcopy(initial_state))
apply_mask(model, mask_dict)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, train_loader, optimizer, criterion, device, epochs=5)
acc_final = test(model, test_loader, device)
print(f"Final winning ticket accuracy: {acc_final:.4f}")
