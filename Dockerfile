# Use the official Python 3.11 image from Docker Hub as the base image
FROM python:3.11

# Copy all files from the current directory on the host to the /app directory in the container
COPY . /app

# Set the working directory to /app inside the container
WORKDIR /app

# Install the Python packages listed in the requirements.txt file
RUN pip install -r requirements.txt

# Set the Flask environment and specify the application entry point
ENV FLASK_APP=app.py

# Expose port 5000 (default Flask port), which can be overridden with the $PORT environment variable
EXPOSE 5000

# Command to run the Flask application, making it accessible on all IP addresses and using the specified port
CMD ["flask", "run", "--host=0.0.0.0", "--port=${PORT:-5000}"]