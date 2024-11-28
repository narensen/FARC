import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import optim

# Label encoder
l_encoder = LabelEncoder()
l_encoder.fit(["OR", "CG"])

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load pre-trained SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load dataset
path = "/home/naren/Documents/Fake Reviews Classifier/data/fake reviews dataset.csv"
df = pd.read_csv(path)
df = df.reset_index(drop=True)

# Dataset class with precomputed embeddings
class FakeReviewsDataset(Dataset):
    def __init__(self, data, labels, device, model, train=True):
        self.data = data
        self.labels = labels
        self.device = device
        self.train = train

        # Precompute embeddings
        self.embeddings = [torch.tensor(model.encode(row["text_"], convert_to_tensor=True)).to(device)
                           for _, row in data.iterrows()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encoded_review = self.embeddings[idx]
        rating = self.data.iloc[idx]["rating"]

        if self.train:
            labels = self.labels[idx]
            labels = torch.tensor(labels).to(self.device)
            rating = torch.tensor(rating).to(self.device)
            return encoded_review, labels, rating
        else:
            rating = torch.tensor(rating).to(self.device)
            return encoded_review, rating

# Train-test split
train = df[["text_", "rating"]]
target = df["label"]

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)
y_train, y_test = l_encoder.fit_transform(y_train), l_encoder.fit_transform(y_test)

# Create datasets and data loaders
train_review = FakeReviewsDataset(X_train, y_train, device, sentence_model)
test_review = FakeReviewsDataset(X_test, y_test, device, sentence_model, train=False)

train_loader = DataLoader(train_review, batch_size=32, shuffle=True)
test_loader = DataLoader(test_review, batch_size=32, shuffle=False)

# Model definition
class FakeReviewClassifier(nn.Module):
    def __init__(self, num_labels, hidden_size=128, num_layers=2, dropout=0.3, input_size=384):  # Default input_size for MiniLM
        super(FakeReviewClassifier, self).__init__()
        
        # Stacked LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0,  # No dropout for a single LSTM layer
                            bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, embeddings):
        
        embeddings = embeddings.unsqueeze(1)  
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embeddings)
        lstm_out_last = lstm_out[:, -1, :] 
        
        # Pass through fully connected layer
        logits = self.fc(lstm_out_last)
        return logits


num_labels = len(l_encoder.classes_)  
hidden_size = 128
num_layers = 2
dropout = 0.3
model = FakeReviewClassifier(num_labels, hidden_size, num_layers, dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for review, label, _ in train_loader:
        review = review.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(review)
        loss = criterion(logits, label)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")