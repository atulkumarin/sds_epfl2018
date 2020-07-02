# Script to build the docker image
docker build -t worker .
docker tag worker aliostux/worker:latest
docker push aliostux/worker
