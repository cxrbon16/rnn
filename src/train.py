import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

from model import RnnEncoder, Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("alibayram/tr_tokenizer")

encoder = RnnEncoder().to(device)
classifier = Classifier().to(device)

ds = load_dataset("emredeveloper/Turkish-Synthetic-Text-Classification")

mapping = {"olumlu": 2, "nötr": 1, "olumsuz": 0}

def handle_label(example):
    return {
        "label": mapping[example["label"]] 
    }

ds = ds.map(handle_label)

train_ds = ds["train"]
test_ds  = ds["test"]


encoder.eval()
classifier.eval()

correct = 0
total = 0

def forward_text(text):
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)

    h = torch.zeros(128, 1).to(device)

    for token_id in token_ids:
        h = encoder(token_id, h)

    logits = classifier(h).view(-1)  # (num_classes,)
    return logits


with torch.no_grad():
    for sample in test_ds:
        text = sample["text"]
        label = int(sample["label"])

        logits = forward_text(text)
        pred = torch.argmax(logits).item()

        if pred == label:
            correct += 1
        total += 1

print("Initial Test Accuracy:", correct / total)

encoder.train()
classifier.train()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    list(encoder.parameters()) + list(classifier.parameters()),
    lr=3e-4
)

epochs = 5

for epoch in range(epochs):
    encoder.train()
    classifier.train()

    total_loss = 0.0

    for sample in train_ds:
        text = sample["text"]
        label = torch.tensor(int(sample["label"])).to(device)

        optimizer.zero_grad()

        logits = forward_text(text)
        loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

encoder.eval()
classifier.eval()

correct = 0
total = 0

with torch.no_grad():
    for sample in test_ds:
        text = sample["text"]
        label = int(sample["label"])

        logits = forward_text(text)
        pred = torch.argmax(logits).item()

        if pred == label:
            correct += 1
        total += 1

print("Test Accuracy:", correct / total)

