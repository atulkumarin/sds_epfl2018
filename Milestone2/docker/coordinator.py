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
import time


def start_(address,_async = True):
    '''Outline :
        load configuration file : ip addresses + port, data path, learning configuration
        Loading Phase:
            send port information to all workers
            send data to all workers
            send learning config signal to all workers
    '''

    coordinator = SVMServicer(is_worker=False)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    nb_workers = int(os.environ['NUMBER_REPLICAS'])
    SVM_pb2_grpc.add_SVMServicer_to_server(coordinator, server)
    coordinator.my_server = server
    server.add_insecure_port(address)
    server.start()

    # Creating stubs for each worker channel
    # Send channel information to each worker
    coordinator.worker_stubs = []
    futures_ = []
    for worker_nb in range(nb_workers - 1):

        worker_name = 'sgd-{}.sgd'.format(worker_nb)
        channel = '{}:50051'.format(socket.gethostbyname(worker_name))
        channel_obj = grpc.insecure_channel(channel)
        stub = SVM_pb2_grpc.SVMStub(channel_obj)
        coordinator.worker_stubs.append(stub)
        grpc.channel_ready_future(channel_obj).result()
        print('[INFO] Worker {} ready'.format(worker_nb))
        msg = SVM_pb2.Node_Config()
        thread = Thread(target=stub.SendNodeInfo, args=(msg,))
        thread.start()
        futures_.append(thread)
        # futures_.append(stub.SendNodeInfo.future(msg))
    [x.join() for x in futures_]
    coordinator.start_time = time.time()

    #[x.result() for x in futures_]
    if _async:
        for worker in coordinator.worker_stubs:
            worker.Start(SVM_pb2.Null())
    else:
        while coordinator.iter_num < coordinator.tot_iter:
            coordinator.update_weight(coordinator.rcv_grads)
            grad_msg = dict_to_weight_msg(coordinator.rcv_grads)
            coordinator.rcv_grads = {}
            grad_msg.iteration_number = coordinator.iter_num
            grad_msg.worker_nb = -1

            futures_ = []
            for worker in coordinator.worker_stubs:
                futures_.append(worker.GetUpdate.future(grad_msg))

            for x in  futures_:
                grad_msg = x.result()
                coordinator.get_weight_helper(grad_msg)
            coordinator.iter_num += 1
    return server


