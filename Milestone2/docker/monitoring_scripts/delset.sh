# Stop all pods and restarts the pods them if not specified otherwise
kubectl delete pods -l app=sgd 2> /dev/null 
kubectl delete statefulset sgd 2> /dev/null 
kubectl delete service sgd 2> /dev/null 
if [ "$1" != "no-start" ]; then
		cmd="kubectl create -f $1"
		eval $cmd
	    
fi

