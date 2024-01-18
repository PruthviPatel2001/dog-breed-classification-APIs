from django.shortcuts import render
# classifier/views.py
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST,require_GET
from PIL import Image
from io import BytesIO
from torchvision import transforms, models
from django.conf import settings
# Import your placeholder model
import torch
import os
from torch import nn
from PIL import Image

dog_breeds =['african_hunting_dog', 'american_staffordshire_terrier', 'australian_terrier', 'basset', 'beagle', 'bernese_mountain_dog', 'border_terrier', 'boston_bull', 'boxer', 'chow', 'doberman', 'english_foxhound', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'golden_retriever', 'great_dane', 'great_pyrenees', 'japanese_spaniel', 'labrador_retriever', 'leonberg', 'mexican_hairless', 'newfoundland', 'norfolk_terrier', 'pomeranian', 'pug', 'rottweiler', 'shih-tzu', 'siberian_husky']
# Load your model
model = models.resnet50(pretrained=True)
num_classes = len(dog_breeds)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the class labels
classes = dog_breeds

# Load the model from the checkpoint
model_path = '/Users/pruthvipatel/Documents/projects/dog_breed_api/dog_breed_model.pth' 
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transformation for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@require_GET
def welcome(request):
    response_data = {'message': "Welcome to Dog Breed Classifier!"}
    return JsonResponse(response_data)

@csrf_exempt
@require_POST
def predict_dog_breed(request):
    # Retrieve image from the request
    image = request.FILES.get('image')
    if not image:
        return HttpResponse("No image found in the request")
    # Preprocess the image
    input_image = Image.open(image).convert('RGB')
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Use the model for prediction
    with torch.no_grad():
        model.eval()
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_label = classes[predicted_idx.item()]

    # Return the result as JSON
    response_data = {'predicted_breed': predicted_label}
    return JsonResponse(response_data)


# Create your views here.
