from concurrent import futures
import grpc
import SVM_pb2_grpc
from server import SVMServicer
import time
import os

# class WokerServicer(SVMServicer):


def serve(address):
    '''starts the server    '''
    try:
        worker = SVMServicer(is_worker=True)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10000000))
        SVM_pb2_grpc.add_SVMServicer_to_server(worker, server)
        worker.my_server = server
        server.add_insecure_port(address)
        server.start()
    except KeyboardInterrupt:
        server.stop(0)

    return server


def start_(address):
    return serve(address)

