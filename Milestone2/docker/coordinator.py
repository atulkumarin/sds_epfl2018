import grpc
from concurrent import futures
import SVM_pb2
import SVM_pb2_grpc
from server import SVMServicer
from utils import *
from time import sleep
import socket
import os
##
import configuration as config
from threading import Thread
##


def start_(address):
    '''Outline :
        load configuration file : ip addresses + port, data path, learning configuration
        Loading Phase:
            send port information to all workers
            send data to all workers
            send learning config signal to all workers
    '''

    coordinator = SVMServicer(is_worker=False)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nb_workers = int(os.environ['NUMBER_REPLICAS'])
    SVM_pb2_grpc.add_SVMServicer_to_server(coordinator, server)
    coordinator.my_server = server
    server.add_insecure_port(address)
    server.start()

    # Creating stubs for each worker channel
    # Send channel information to each worker
    coordinator.worker_stubs = []
    futures_ = []
    print(range(nb_workers-1))
    for worker_nb in range(nb_workers-1):

        worker_name = 'sgd-{}.sgd'.format(worker_nb)
        channel = '{}:50051'.format(socket.gethostbyname(worker_name))
        channel_obj = grpc.insecure_channel(channel)
        stub = SVM_pb2_grpc.SVMStub(channel_obj)
        coordinator.worker_stubs.append(stub)

        msg = SVM_pb2.Node_Config()
        #msg.coordinator_address = '{}:50051'.format(os.environ['MY_POD_IP'])

        #to_send = config.channels.copy()

        #msg.workers_address.extend(to_send)
        #msg.worker_nb = idx

        grpc.channel_ready_future(channel_obj).result()
        print('[INFO] Worker {} ready'.format(worker_nb))
        thread = Thread(target=stub.SendNodeInfo,args=(msg,))
        thread.start()
        futures_.append(thread)
        #futures_.append(stub.SendNodeInfo.future(msg))

    #[x.result() for x in futures_]
    [x.join() for x in futures_]

    for worker in coordinator.worker_stubs:
        worker.Start(SVM_pb2.Null())
    return server
