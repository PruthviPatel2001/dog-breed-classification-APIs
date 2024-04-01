from torchvision import transforms,models
from torch import nn
from PIL import Image
import torch
from dogdata.dog_breed_data import dog_breeds_data, dog_breeds



def load_model(model_path, num_classes, dog_breeds):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = Image.open(image).convert('RGB')
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def find_dog_breed(input_batch, model):
    classes = dog_breeds
    with torch.no_grad():
        model.eval()
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = classes[predicted_idx.item()]
    probabilities = torch.softmax(output, dim=1)[0] * 100
    return predicted_label, probabilities[predicted_idx.item()]

def get_breed_details(breed_name):
    breed_details = next((breed for breed in dog_breeds_data if breed['breed_name'] == breed_name), None)
    return breed_details