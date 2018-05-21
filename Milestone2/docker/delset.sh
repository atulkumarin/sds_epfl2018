kubectl delete pods --all
kubectl delete statefulset sgd
kubectl delete service sgd
docker build -t worker .
kubectl create -f conf.yaml 
