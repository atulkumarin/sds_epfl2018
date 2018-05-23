import configuration
from worker import Worker
from coordinator import Coordinator
import time


servers = []
try:
    for channel in configuration.channels:
        worker = Worker()
        servers.append(worker.start_(channel))
    print('[INFO] LOADED ALL WORKERS')

    coordinator = Coordinator()
    c = coordinator.start_(configuration.coordinator_channel, servers)
    print('[INFO] LOADED COORD')
    
    coordinator.start_train(c)

    while(True):
        time.sleep(3600 * 24)
except KeyboardInterrupt:
    print('[ERROR] EXIT')
    [x.stop(0) for x in servers]
