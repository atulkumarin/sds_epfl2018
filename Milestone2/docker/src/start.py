import worker
import coordinator
import time
import os

# Entry script

try:
    # check if the current pod needs to be a worker or a coordinator and start it
    pod_nb = int(os.environ['MY_POD_NAME'].split('-')[1])
    number_pods = int(os.environ['NUMBER_REPLICAS'])
    my_channel = os.environ['MY_POD_IP'] + ':50051'
    _async = True if (os.environ['ASNYCH'] == '1') else False
    if _async:
        print('[INFO] Asynchronous mode')
    else:
        print('[INFO] Synchronous mode')
    if(pod_nb == number_pods - 1):
        print('[INFO] Coordinator started')
        server = coordinator.start_(
            my_channel, _async=_async)
    else:
        print('[INFO] Worker {} started'.format(pod_nb))
        server = worker.start_(my_channel)

    while(True):
        time.sleep(3600 * 24)
except KeyboardInterrupt:
    print('[ERROR] EXIT')
