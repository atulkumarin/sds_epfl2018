import configuration
import worker
import coordinator
import worker
import time
import os


servers = []
try:
    pod_nb = int(os.environ['MY_POD_NAME'].split('-')[1])
    number_pods = int(os.environ['NUMBER_REPLICAS'])
    my_channel = os.environ['MY_POD_IP']+':50051'

    if(pod_nb == number_pods - 1):
        server = coordinator.start_(my_channel)
        print('[INFO] Coordinator started')
    else:
        server = worker.start_(my_channel)
        print('[INFO] Worker {} started'.format(pod_nb))

    while(True):
        time.sleep(3600 * 24)
except KeyboardInterrupt:
    print('[ERROR] EXIT')
    [x.stop(0) for x in servers]
