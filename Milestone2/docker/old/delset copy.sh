# Stop all pods and restarts them if not specified otherwise
kubectl delete pods --all
kubectl delete statefulset sgd
kubectl delete service sgd
if [ "$2" != "no-start" ]; then
	if [ "$1" = "cluster" ]; then
	    kubectl create -f kubernetes/conf_cluster.yaml 
	else
	    kubectl create -f kubernetes/conf.yaml 
	fi
fi

