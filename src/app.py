import streamlit as st
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision import datasets

# ==================================
# Page Config
# ==================================

st.set_page_config(
    page_title="Car AI Classifier",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car AI Classifier")

st.markdown("""
Upload a car image and the AI will identify the vehicle model.

### Supported Brands
- Audi
- BMW
- Ford
- Mercedes-Benz
- Toyota

### Supported Models
49 Vehicle Models
""")

# ==================================
# Device
# ==================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================
# Classes
# ==================================

dataset = datasets.ImageFolder("dataset/train")
classes = dataset.classes

# ==================================
# Load Model
# ==================================

@st.cache_resource
def load_model():

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

    return model


model = load_model()

# ==================================
# Image Transform
# ==================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================
# Upload
# ==================================

uploaded_file = st.file_uploader(
    "Choose a car image",
    type=["jpg", "jpeg", "png"]
)

# ==================================
# Prediction
# ==================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        top_probs, top_indices = torch.topk(
            probs,
            k=3
        )

    best_idx = top_indices[0][0].item()
    best_prob = top_probs[0][0].item() * 100

    st.success("Prediction Complete ✅")

    # ===============================
    # Main Prediction
    # ===============================

    st.subheader("🏆 Predicted Model")

    st.write(
        f"### {classes[best_idx]}"
    )

    st.write(
        f"Confidence: **{best_prob:.2f}%**"
    )

    st.progress(best_prob / 100)

    # ===============================
    # Confidence Level
    # ===============================

    if best_prob >= 90:

        st.success(
            "High Confidence ✅"
        )

    elif best_prob >= 70:

        st.warning(
            "Medium Confidence ⚠️"
        )

    else:

        st.error(
            "Low Confidence ❌"
        )

    # ===============================
    # Top 3 Predictions
    # ===============================

    st.subheader("📊 Top 3 Predictions")

    for i in range(3):

        idx = top_indices[0][i].item()

        prob = (
            top_probs[0][i].item() * 100
        )

        st.write(
            f"**{i+1}. {classes[idx]}** — {prob:.2f}%"
        )
