kubectl delete pods --all
kubectl delete statefulset sgd
kubectl delete service sgd
#docker build -t worker .
if [ "$2" != "no-start" ]; then
	if [ "$1" = "cluster" ]; then
	    kubectl create -f conf_cluster.yaml 
	else
	    kubectl create -f conf.yaml 
	fi
fi

