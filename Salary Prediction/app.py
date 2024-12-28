from flask import Flask, render_template, request, jsonify
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (named LR)
with open('salary_LR.pkl', 'rb') as file:
    LR = pickle.load(file) # Load the model and assign it to the variable LR

# Route to serve the HTML file
@app.route('/')
def home():
    return render_template('index.html') # Serve index.html from the 'templates' folder

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON input from the frontend
    data = request.json
    years_of_experience = float(data['years_of_experience']) # Extract years of experience from input
    prediction = LR.predict([[years_of_experience]]) # Predict salary using the model
    return jsonify({'predicted_salary': prediction[0]}) # Return the predicted salary as JSON

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)