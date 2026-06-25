import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

# ======================
# Classes
# ======================

from torchvision import datasets

dataset = datasets.ImageFolder("dataset/train")
classes = dataset.classes

# ======================
# Device
# ======================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================
# Model
# ======================

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "best_model1.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ======================
# Transform
# ======================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Image
# ======================

image = Image.open("test3.jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(DEVICE)

# ======================
# Predict
# ======================

with torch.no_grad():

    outputs = model(image)

    probs = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(
        probs,
        k=3
    )

print("\nTop 3 Predictions:\n")

for i in range(3):

    idx = top_indices[0][i].item()

    prob = top_probs[0][i].item() * 100

    print(
        f"{i+1}. {classes[idx]} - {prob:.2f}%"
    )