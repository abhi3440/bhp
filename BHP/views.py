from django.shortcuts import render
import joblib
import numpy as np

# Create your views here.

def home(request):
    """
    Render the home page.
    """
    return render(request, 'home.html')

def predict(request):
    """
    Render the prediction page.
    """
    return render(request, 'predict.html')

# Assuming model.joblib is saved in the same directory as views.py or in a known location
MODEL_FILE_PATH = "linear_regression.joblib"
le = joblib.load('label_encoder.pkl')
def result(request):
    """
    Render the result page.
    """
    location = request.GET.get('n1')
    total_sqft = float(request.GET.get('n2'))
    bath = int(request.GET.get('n3'))
    bhk = int(request.GET.get('n4'))
    le = joblib.load('label_encoder.pkl')
    location_encoded = le.transform([location])[0]
    # Load the model
    model = joblib.load(MODEL_FILE_PATH)
    input_data = np.array([[location_encoded,total_sqft, bath, bhk]])
    predicted_price = model.predict(input_data)

    context = {
        'predicted_price': predicted_price[0],  # Assuming a single prediction
    }
    return render(request, 'predict.html', context)


