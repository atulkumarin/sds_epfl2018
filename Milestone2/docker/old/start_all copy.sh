#Script that starts the training and kills all the pods when training is finished or when users press crtl+c
control_c() {
./delset.sh cluster no-start
echo '[INFO] EXIT'
exit
}

trap control_c SIGINT

chmod +x delset.sh
./delset.sh cluster



coordinator_id=$(( $1 -1 ))
cond='NOK'
while [ $cond != 'OK' ]; do
	{ # try
		cmd="kubectl logs sgd-$coordinator_id 2> /dev/null >/dev/null"
	    eval $cmd && cond='OK'
	    
	    #save your output

	} || { # catch
	     cond='NOK'
	}
done
echo "[INFO] all workers started"
cmd="kubectl logs -f sgd-$coordinator_id > ../logs.txt &"
eval $cmd 
pid_log=$!


if [ "$2" == 'dynamic_plot' ]; then
	cmd="python monitor_log.py --nb_workers $coordinator_id --dynamic_plot --log_file ../logs.txt --nb_iter $3 "
	eval $cmd
else
	cmd="python monitor_log.py --nb_workers $coordinator_id --log_file logs.txt"
	eval $cmd
fi
echo "ok"
cmd='kill $pid_log'
eval $cmd 
cmd="kubectl cp sgd-$coordinator_id:/data/weights.json ."
eval $cmd
./delset.sh cluster no-start