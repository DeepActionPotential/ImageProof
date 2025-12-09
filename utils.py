import torch
import timm
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 🔹 Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(model_path: str):
    """Load EfficientNet-B3 model from file."""
    if model_path.endswith(".pt"):
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    else:
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_image(input_source):
    """Load and preprocess image from file upload or URL."""
    if isinstance(input_source, str):  # URL
        response = requests.get(input_source)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:  # Uploaded file
        img = Image.open(input_source).convert("RGB")
    return img


def preprocess_image(img):
    """Apply transform and return tensor."""
    return transform(img).unsqueeze(0)


def predict(model, img_tensor):
    """Predict class and confidence score."""
    with torch.no_grad():
        outputs = model(img_tensor.to(DEVICE))
        probs = torch.sigmoid(outputs)
        prob = probs.item()
        print(prob)
        label = "🧠 AI-generated" if prob >= 0.00001 else "📸 Real"
    return label, prob
