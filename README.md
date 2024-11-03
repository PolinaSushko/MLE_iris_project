# Iris Classification Project
This project is a machine learning pipeline for training and inferring with a deep learning model on the Iris dataset. The entire workflow is Dockerized, from data preprocessing to model training and inference.

# Project Structure
```
├── artifacts/                       # Stores output files like trained model and preprocessed data artifacts
├── logs/                            # Logs for tracking system and model activities
├── notebook/                        # Exploratory Data Analysis (EDA) and prototyping the model training process notebooks
│   ├── EDA.ipynb                
│   ├── model_training.ipynb                      
├── src/                             # Source code directory containing core project modules
│   ├── components/                  # Core modules for each stage of the machine learning pipeline
│   |    ├── data_ingestion.py 
│   |    ├── data_transformation.py        
│   |    ├── model_trainer.py
│   ├── pipeline/                    # Pipeline for running predictions
│   |    ├── predict_pipeline.py  
|   ├── exception.py                 # Custom exception handling for error tracking
|   ├── logger.py                    # Logging setup for monitoring and debugging
|   ├── utils.py                     # Utility functions for data processing and modeling tasks
├── templates/                       # HTML templates for the web application interface
|   ├── home.html
|   ├── index.html
├── .gitignore  
├── app.py                           # Main Flask application file for serving the web app
├── Dockerfile                       # Dockerfile to containerize the application
├── README.md   
├── requirements.txt                 # List of required packages for the project          
└── setup.py                         # Setup script for packaging and distribution of the project
```

# Setup Instructions
## Softaware and tools requirements
1. [GitHub Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/)
3. [Docker](https://www.docker.com/)

## Installation
1. Clone this repository:
```
git clone https://github.com/PolinaSushko/MLE_iris_project.git
cd MLE_iris_project
```
2. Create a new virtual environment:
```
python -m venv venv
```
3. Activate the virtual environment:
```
venv\Scripts\activate
```
4. Build Docker image for app display:
```
docker build -t iris_app .
```

## Usage
Use `iris_app` to run the app:
```
docker run -p 5000:5000 iris_app
```

# Error Handling
Exceptions manage issues like missing model files or data preprocessing errors, with logs for monitoring performance and dataset size.

# Logging
Logs display model performance at each step.

