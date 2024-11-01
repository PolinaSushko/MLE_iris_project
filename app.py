from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

applictaion = Flask(__name__)

app = applictaion

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_data():
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
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
