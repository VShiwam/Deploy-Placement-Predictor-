from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        student_iq = request.form.get('student_iq')
        cgpa = float(request.form.get('cgpa'))
        
        # Prepare input for prediction
        input_features = np.array([[float(student_iq), cgpa]])  # Ensure both features are included
        prediction = model.predict(input_features)
        
        # Assuming model outputs 1 for placement and 0 for no placement
        result = "Placement Hoga" if prediction[0] == 1 else "Placement Nahi Hoga"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
