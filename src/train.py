import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader

# ======================
# Config
# ======================

BATCH_SIZE = 8
IMG_SIZE = 224
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", DEVICE)

# ======================
# Transforms
# ======================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Dataset
# ======================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=val_transform
)

print("Classes:", len(train_dataset.classes))
print("Train Images:", len(train_dataset))
print("Test Images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ======================
# Model
# ======================

model = efficientnet_b0(weights=None)

# نفس عدد كلاسات موديل البراندات القديم
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    33
)

# تحميل الأوزان القديمة
state_dict = torch.load(
    r"C:\Users\abdou\Desktop\future_projects\DL\Car\best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(state_dict)

print("Brand model loaded successfully!")

# change last layer with 49 class

num_classes = len(train_dataset.classes)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(DEVICE)

# ======================
# Loss + Optimizer
# ======================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ======================
# Training
# ======================

best_acc = 0.0

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ======================
    # Validation
    # ======================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * correct / total

    scheduler.step(val_acc)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            f"New Best Model Saved! "
            f"Accuracy = {best_acc:.2f}%"
        )

print("\nTraining Finished!")
print(
    f"Best Validation Accuracy: "
    f"{best_acc:.2f}%"
)
