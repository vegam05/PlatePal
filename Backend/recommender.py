import requests
import json
from transformers import pipeline

# Dummy User Health Profile
USER_PROFILE = {
    "age": 25,
    "gender": "male",
    "weight": 70,  # kg
    "height": 175,  # cm
    "activity_level": "moderate",  # Options: sedentary, moderate, active
    "health_goals": ["weight loss"],  # Options: weight loss, maintenance, weight gain
    "dietary_restrictions": ["low sugar"],  # E.g., vegetarian, low sugar, gluten-free
}

# API Keys (Replace with your actual keys)
FOODVISOR_API_KEY = "APIkey"
# Initialize the pipeline (this should ideally be done outside the function for efficiency)
text_generation_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")


# Foodvisor API URL
FOODVISOR_API_URL = "https://api.foodvisor.io/v1/food-recognition"



# Upload Image and Get Food Recognition Data
def recognize_food(image_path):
    headers = {"Authorization": f"Bearer {FOODVISOR_API_KEY}"}
    files = {"file": open(image_path, "rb")}
    
    response = requests.post(FOODVISOR_API_URL, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Foodvisor API Error: {response.text}")

def generate_analysis(food_data, user_profile):
    prompt = f"""
You are a health assistant. A user has provided the following food data:
{json.dumps(food_data, indent=2)}

User's health profile:
{json.dumps(user_profile, indent=2)}

Based on this information:
1. Provide a nutritional analysis of the food.
2. Suggest whether this food aligns with the user's health goals and dietary restrictions.
3. Give specific recommendations for the user.
"""
    # Generate the output using the pipeline
    output = text_generation_pipeline(prompt, max_length=500, do_sample=True)[0]["generated_text"]
    return output

# Main Function
def main(image_path):
    try:
        # Step 1: Recognize Food
        print("Recognizing food in the image...")
        food_data = recognize_food(image_path)
        print("Food recognition data received.")

        # Step 2: Generate Nutritional Analysis and Recommendations
        print("Generating nutritional analysis and recommendations...")
        analysis = generate_analysis(food_data, USER_PROFILE)
        print("Analysis and recommendations:")
        print(analysis)
    
    except Exception as e:
        print(f"Error: {e}")

# Run the Script
if __name__ == "__main__":
    # Provide the path to the image
    image_path = "path_to_your_image.jpg"
    main(image_path)
