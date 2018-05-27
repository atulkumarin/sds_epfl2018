#Script that starts the training and kills all the pods when training is finished or when users press crtl+c
control_c() {
echo '[INFO] Started deleting active pods'
monitoring_scripts/delset.sh no-start
echo '[INFO] EXIT'
exit
}

usage() {
  echo -n "
 
 This script starts the training process and saves the log file in 'log.txt' and the training plots on a folder named 'plot'

 Options:

  -d, --dynamic_plot     enable dynamic plotting mode to monitor the training	
  -i, --nb_iter          Number of iterations (by default 1000),this value needs to be equal to one in the kubernetes config file
  -h, --help             Display this help and exit
  -w, --nb_pods          Number of pods (by default 4),this value needs to be equal to one in the kubernetes config file


"
}
coordinator_id=3
cond='NOK'
nb_iter=1000
dynamic_plot=false
while true; do
  case "$1" in
    -h | --help )    usage >&2; exit ;;
    -i | --nb_iter ) nb_iter="$2"; shift; shift ;;
    -w | --nb_pods ) coordinator_id=$(($2 -1)); shift; shift ;;
	-d | --dynamic_plot ) dynamic_plot=true; shift;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

trap control_c SIGINT

chmod +x monitoring_scripts/delset.sh
monitoring_scripts/delset.sh monitoring_scripts/tmp.yaml



#coordinator_id=$(( $1 -1 ))

while [ $cond != 'OK' ]; do
	sleep .5
	{ 
		cmd="kubectl logs sgd-$coordinator_id 2> /dev/null >/dev/null"
	    eval $cmd && cond='OK'
	    
	    #save your output

	} || { # catch
	     cond='NOK'
	}
done
echo "[INFO] all workers started"
cmd="kubectl logs -f sgd-$coordinator_id > logs.txt &"
eval $cmd 
pid_log=$!


if [ "$dynamic_plot" == 'true' ]; then
	cmd="python monitoring_scripts/monitor_log.py --nb_workers $coordinator_id --dynamic_plot --log_file logs.txt --nb_iter $nb_iter "
	eval $cmd
else
	echo '[INFO] Dynamic plotting mode OFF'
	cmd="python monitoring_scripts/monitor_log.py --nb_workers $coordinator_id --log_file logs.txt --nb_iter $nb_iter "
	eval $cmd
fi

cmd="kill $pid_log"
eval $cmd 
cmd="kubectl cp sgd-$coordinator_id:/data/weights.json ."
eval $cmd
echo '[INFO] Started deleting active pods'
monitoring_scripts/delset.sh no-start
echo "[INFO] Aborted or ended training and sucessfully deleted all active pods"
