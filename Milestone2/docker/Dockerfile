# Use an official Python runtime as a parent image
FROM python:3.6

# Creating Application Source Code Directory



# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app


# Install vim for quick edits
RUN apt-get update
RUN apt-get -y install vim

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r /app/requirements.txt

# VOLUME ["/Users/alialami-idrissi/Documents/MA2/ENVS"]
# Run app.py when the container launches
ENTRYPOINT ["python","-u", "/app/dummy.py"]