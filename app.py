from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

# Load the training data (or load it from a CSV file or wherever your data is stored)
train_data = pd.read_csv('artifacts/raw.csv')  # Adjust path as needed
locations = train_data['location'].unique()  # Get unique locations

# Route for the home page (initial page with a message)
@app.route("/")
def home():
    # This route will render home.html when the user visits the root URL
    return render_template('home.html')

# Route for handling form submission and making predictions (accessible via /predictdata)
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Get data from the form
        data = CustomData(
            location=request.form.get('location'),
            total_sqft=request.form.get('area'),
            bath=request.form.get('bath'),
            bhk=request.form.get('bhk')
        )

        # Prepare data for prediction
        pred_data = data.get_data_as_data_frame()
        print(pred_data)

        # Run the prediction pipeline
        predic_pipeline = PredictPipeline()
        result = predic_pipeline.predict(pred_data)

        # Render the result page (index.html) with the prediction result
        return render_template('index.html', locations=locations, result=result[0])  # Show result in index.html

    else:
        # If the request is a GET request, render the form
        return render_template('index.html', locations=locations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
