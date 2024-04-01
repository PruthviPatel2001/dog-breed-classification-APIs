
# classifier/views.py
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST,require_GET
import os
from dogdata.dog_breed_data import dog_breeds
from classifier_utlis.utlis import load_model, preprocess_image, find_dog_breed, dog_breeds,get_breed_details


# Load the model from the checkpoint
base_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up from the current directory
project_dir = os.path.dirname(base_dir)

# Load the model from the checkpoint
model_path =  os.path.join(project_dir, 'dog_breed_model.pth')
num_classes = len(dog_breeds)
model = load_model(model_path, num_classes, dog_breeds)

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
    input_batch = preprocess_image(image)
    
    # Get the predicted breed and confidence score
    predicted_breed, confidence_score = find_dog_breed(input_batch, model)

    # Get the breed details
    breed_details = get_breed_details(predicted_breed)

    response_data = {
        'predicted_breed': predicted_breed,
        'breed_details': breed_details,
        'confidence_score': round(float(confidence_score), 2)
    }

    return JsonResponse(response_data)



