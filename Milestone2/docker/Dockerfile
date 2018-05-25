# Use an official Python runtime as a parent image
FROM python:3.6

# Creating Application Source Code Directory



# Set the working directory to /app
WORKDIR /app


# Install vim for quick edits
RUN apt-get update
RUN apt-get -y install vim


# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

ADD . /app

RUN pip install -r /app/requirements.txt

# VOLUME ["/Users/alialami-idrissi/Documents/MA2/ENVS"]

# Copy the current directory contents into the container at /app
# Run app.py when the container launches
ENTRYPOINT ["python","-u", "/app/dummy.py"]
#ENTRYPOINT ["python","-u", "/../start.py"]