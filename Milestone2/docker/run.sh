#Script that starts the training and kills all the pods when training is finished or when users press crtl+c
control_c() {
echo '[INFO] Started deleting active pods'
cmd='cd $pwd_var'
eval $cmd
cmd='kill $pid_log'
eval $cmd 2> /dev/null
rm -rf monitoring_scripts/tmp.yaml
monitoring_scripts/delset.sh no-start
echo '[INFO] EXIT'
exit
}

usage() {
  echo -n "
 
 This script starts the training process,saves the log file in 'log.txt',the training plots on a folder named 'plot' and the weight vector in 'weights.json'

 Options:

  -d, --dynamic_plot     enable dynamic plotting mode to monitor the training
  -asynch, --asynch      enable asynch mode,if not specified the config file value will be used
  -synch, --synch        enable synch mode,if not specified the config file value will be used

"
}
coordinator_id=3
cond='NOK'
dynamic_plot=false
synch=''
while true; do
  case "$1" in
    -h | --help )    usage >&2; exit ;;
	-d | --dynamic_plot ) dynamic_plot=true; shift;;
	-synch | --synch ) synch='--synch'; shift;;
	-asynch | --asynch ) synch='--asynch'; shift;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done
pwd_var=`pwd`
trap control_c SIGINT

cd monitoring_scripts

coordinator_id=$(python -u parse_conf.py $synch)

coordinator_id=$(($coordinator_id -1))
chmod +x delset.sh
./delset.sh tmp.yaml


cd ../

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



cd monitoring_scripts
if [ "$dynamic_plot" == 'true' ]; then
	cmd="python monitor_log.py --dynamic_plot --log_file ../logs.txt "
	eval $cmd
else
	echo '[INFO] Dynamic plotting mode OFF'
	cmd="python monitor_log.py  --log_file ../logs.txt "
	eval $cmd
fi
cd ../

cmd="kill $pid_log"
eval $cmd 
cmd="kubectl cp sgd-$coordinator_id:/data/weights.json ."
eval $cmd
echo '[INFO] Started deleting active pods'
monitoring_scripts/delset.sh no-start
rm -rf monitoring_scripts/tmp.yaml 
echo "[INFO] Aborted or ended training and sucessfully deleted all active pods"

