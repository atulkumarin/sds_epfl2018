import worker
import coordinator
import time
import os


servers = []
print(int(os.environ['ASNYCH']))
try:
    pod_nb = int(os.environ['MY_POD_NAME'].split('-')[1])
    number_pods = int(os.environ['NUMBER_REPLICAS'])
    my_channel = os.environ['MY_POD_IP'] + ':50051'
    _async = True if (os.environ['ASNYCH'] == '1') else False
    print(_async)
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
    [x.stop(0) for x in servers]
