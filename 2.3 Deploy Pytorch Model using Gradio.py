import os
import cv2
import gradio as gr
from PIL import Image
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = nn.Identity()
resnet_model.eval()

fc_model = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1)
)

model = nn.Sequential(
    resnet_model,
    fc_model
)

state_dict = torch.load("tyre.pth", map_location="cuda")
model.load_state_dict(state_dict)
model.eval()     

def predict(pixels):
    image= Image.fromarray(pixels)
    img = preprocess(image).unsqueeze(dim=0)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        predictions = torch.sigmoid(outputs)[0]
        print(predictions.item())
    return predictions.item()

demo = gr.Interface(fn=predict, inputs=[gr.Image()], outputs="text")
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=False
)

