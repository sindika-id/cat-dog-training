import argparse, torch
from PIL import Image
from torchvision import transforms, models
from torch import nn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to image file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/catdog_model.pth", map_location=device))
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(args.img).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()

    labels = ["cat", "dog"]
    print("Prediction:", labels[pred])

if __name__ == "__main__":
    main()