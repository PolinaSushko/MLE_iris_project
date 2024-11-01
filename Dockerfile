# Use the official Python 3.11 image from Docker Hub as the base image
FROM python:3.11

# Copy all files from the current directory on the host to the /app directory in the container
COPY . /app

# Set the working directory to /app inside the container
WORKDIR /app

# Install the Python packages listed in the requirements.txt file
RUN pip install -r requirements.txt

# Expose the port defined by the PORT environment variable to allow external access
EXPOSE $PORT

# Command to run the Flask application, making it accessible on all IP addresses and using the specified port
CMD ["flask", "run", "--host=0.0.0.0", "--port=$PORT"]
