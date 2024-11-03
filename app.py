import os
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

applictaion = Flask(__name__)

app = applictaion

# Route for a home page
@app.route('/')
def index():
    """
    Serves the main page of the web application. This route is accessed via the root URL ('/').
    It renders and returns the 'index.html' template, which typically serves as the homepage 
    for users to interact with the application.

    Returns:
        Response: The HTML content of the 'index.html' template.
    """
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_data():
    """
    Handles requests to the '/predictdata' route, allowing users to input data for predictions.
    Supports both GET and POST requests:
    
    - For a GET request, renders the 'home.html' template, providing a form for user input.
    - For a POST request:
        - Extracts form data submitted by the user, including sepal and petal dimensions.
        - Creates an instance of the CustomData class, which encapsulates the input data and
          converts it into a DataFrame for processing.
        - Initiates the PredictPipeline class to load the model and preprocessor and to make 
          predictions based on the input data.
        - Passes the prediction result back to the 'home.html' template, displaying it to the user.

    Returns:
        Response: Renders 'home.html' either with a blank form (for GET) or with prediction results (for POST).
    """
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            sepal_length = float(request.form.get('sepal_length')),
            sepal_width  = float(request.form.get('sepal_width')),
            petal_length = float(request.form.get('petal_length')),
            petal_width  = float(request.form.get('petal_width'))
        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)

        predict_data = PredictPipeline()
        results      = predict_data.predict(pred_df)

        return render_template('home.html', results = results[0])

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = int(os.environ.get('PORT', 5000)))
