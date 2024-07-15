import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Define the BoxDataset class
class BoxDataset(Dataset):
    def __init__(self, coord_dir, index_dir):
        self.coord_dir = coord_dir
        self.index_dir = index_dir
        self.coord_files = sorted([f for f in os.listdir(coord_dir) if f.endswith('.txt')])
        self.index_files = sorted([f for f in os.listdir(index_dir) if f.endswith('.txt')])

    def __len__(self):
        return len(self.coord_files)

    def __getitem__(self, idx):
        coord_path = os.path.join(self.coord_dir, self.coord_files[idx])
        index_path = os.path.join(self.index_dir, self.index_files[idx])

        # Load coordinates
        with open(coord_path, 'r') as f:
            coords = [list(map(float, line.strip().split())) for line in f]

        # Load indices
        with open(index_path, 'r') as f:
            indices = [int(line.strip()) for line in f]

        coords = torch.tensor(coords, dtype=torch.float32)
        indices = torch.tensor(indices, dtype=torch.long)

        return coords, indices

# Define the collate function
def collate_fn(batch):
    coords, indices = zip(*batch)
    coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=-1)
    indices = torch.nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=-1)
    mask = (indices != -1)
    return coords, indices, mask

# Define the PointerNetwork class
class PointerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_len):
        super(PointerNetwork, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.pointer = nn.Linear(hidden_dim, max_len)

    def forward(self, x, mask):
        enc_out, _ = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        pointer_logits = self.pointer(dec_out)
        pointer_logits[~mask] = float('-inf')  # Masking the padded positions
        return pointer_logits

# Define the training function
def train(model, dataloader, criterion, optimizer, device, pad_idx):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for coords, indices, mask in dataloader:
        coords, indices = coords.to(device), indices.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(coords, mask)
        outputs = outputs.view(-1, outputs.size(-1))
        
        indices = indices.view(-1)
        loss = criterion(outputs, indices)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == indices).sum().item()
        total += indices.size(0)

    return total_loss / len(dataloader), correct / total

# Define the validation function
def validate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for coords, indices, mask in dataloader:
            coords, indices = coords.to(device), indices.to(device)
            mask = mask.to(device)

            outputs = model(coords, mask)
            outputs = outputs.view(-1, outputs.size(-1))
            
            indices = indices.view(-1)
            loss = criterion(outputs, indices)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == indices).sum().item()
            total += indices.size(0)

    return total_loss / len(dataloader), correct / total


# Define the plot_boxes function
def plot_boxes(coords, img_width, img_height):
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.invert_yaxis()

    for i, coord in enumerate(coords):
        rect = patches.Polygon([(coord[0], coord[1]), (coord[2], coord[3]), (coord[4], coord[5]), (coord[6], coord[7])], 
                               closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(coord[0], coord[1], str(i), color='blue', fontsize=12, verticalalignment='top')
    
    plt.show()

# Define the visualization function
def visualize_predictions(model, dataloader, device):
    model.eval()
    all_samples = []
    for coords, indices, mask in dataloader:
        for i in range(len(coords)):
            all_samples.append((coords[i], indices[i], mask[i]))
    
    sample = random.choice(all_samples)
    coords, indices, mask = sample
    coords, indices, mask = coords.to(device), indices.to(device), mask.to(device)

    with torch.no_grad():
        outputs = model(coords.unsqueeze(0), mask.unsqueeze(0))
        pred_indices = outputs.argmax(dim=2).squeeze(0).cpu().numpy()
        true_indices = indices.cpu().numpy()

    coords = coords.cpu().numpy()

    # Remove padding
    mask = pred_indices != -1
    valid_indices = pred_indices[mask]
    valid_coords = coords[mask]

    # Sort coordinates by true indices
    sorted_indices = np.argsort(valid_indices)

    # Sort coordinates by true indices
    sorted_coords = valid_coords[sorted_indices]

    # Plot true boxes
    plot_boxes(sorted_coords, np.max(sorted_coords)*2, np.max(sorted_coords)*2)



# Main function
def main(data_dir, batch_size=256, num_epochs=10, hidden_dim=256, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data directories
    train_coord_dir = os.path.join(data_dir, 'train', 'coordinates')
    train_index_dir = os.path.join(data_dir, 'train', 'indices')
    val_coord_dir = os.path.join(data_dir, 'val', 'coordinates')
    val_index_dir = os.path.join(data_dir, 'val', 'indices')
    test_coord_dir = os.path.join(data_dir, 'test', 'coordinates')
    test_index_dir = os.path.join(data_dir, 'test', 'indices')

    # Load data
    train_dataset = BoxDataset(train_coord_dir, train_index_dir)
    val_dataset = BoxDataset(val_coord_dir, val_index_dir)
    test_dataset = BoxDataset(test_coord_dir, test_index_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = 8  # Dimension of input data (box coordinates)
    pad_idx = -1  # Padding index

    # Initialize model, criterion, and optimizer
    model = PointerNetwork(input_dim=input_dim, hidden_dim=hidden_dim, max_len=batch_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, pad_idx)
        val_loss, val_acc = validate(model, val_loader, criterion, device, pad_idx)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))

    # Test the model
    test_loss, test_acc = validate(model, test_loader, criterion, device, pad_idx)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Visualize predictions
    visualize_predictions(model, test_loader, device)

if __name__ == "__main__":
    data_dir = r"C:\Users\user\Desktop\examples\data"  # Specify the path to the base data folder
    main(data_dir, batch_size=1024, num_epochs=500)
