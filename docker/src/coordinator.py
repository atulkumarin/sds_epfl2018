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


def start_(address, _async=True):
    '''Outline :
        itinialize coordinator and starts the training process
    '''

    # Start coordinator
    coordinator = SVMServicer(is_worker=False)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    nb_workers = int(os.environ['NUMBER_REPLICAS'])
    SVM_pb2_grpc.add_SVMServicer_to_server(coordinator, server)
    coordinator.my_server = server
    server.add_insecure_port(address)
    server.start()

    # Creating stubs for each worker channel
    # Send discovery signal to all other workers
    coordinator.worker_stubs = []
    futures_ = []
    for worker_nb in range(nb_workers - 1):

        worker_name = 'sgd-{}.sgd'.format(worker_nb)
        channel = '{}:50051'.format(socket.gethostbyname(worker_name))
        channel_obj = grpc.insecure_channel(channel)
        stub = SVM_pb2_grpc.SVMStub(channel_obj)
        coordinator.worker_stubs.append(stub)
        # future to test connectivity
        futures_.append(grpc.channel_ready_future(channel_obj))

    # Block until all workers are up
    [x.result() for x in futures_]
    futures_ = []

    # Send signal to every to start looking for other workers
    for stub in coordinator.worker_stubs:
        thread = Thread(target=stub.SendNodeInfo, args=(SVM_pb2.Null(),))
        thread.start()
        futures_.append(thread)

    [x.join() for x in futures_]

    coordinator.start_time = time.time()

    if _async:
        # In asynch mode the coordinator send a start signal to all workers
        for worker in coordinator.worker_stubs:
            coordinator.pool_threads.submit(worker.Start,SVM_pb2.Null())
    else:
        # In synch mode the coordinator synchronizes the workers and takes care of the main training loop
        while coordinator.iter_num < coordinator.tot_iter:
            # Do the optimization step locally
            coordinator.update_weight(coordinator.rcv_grads)
            grad_msg = dict_to_weight_msg(coordinator.rcv_grads)
            coordinator.rcv_grads = {}
            grad_msg.iteration_number = coordinator.iter_num
            grad_msg.worker_nb = -1

            futures_ = []
            # Send accumulated gradient and Start computing gradients at all workers
            for worker in coordinator.worker_stubs:
                futures_.append(worker.GetUpdate.future(grad_msg))
            # Block until all workers sent back the gradients and accumulate them
            for x in futures_:
                grad_msg = x.result()
                coordinator.get_weight_helper(grad_msg)
            coordinator.iter_num += 1
    return server
