from concurrent import futures
import grpc
import sys
import SVM_pb2
import SVM_pb2_grpc
import random
from utils import *
from server import SVMServicer
import time


#class WokerServicer(SVMServicer):

class Worker():
    def serve(self,address):
        '''starts the server    '''
        try:
            worker = SVMServicer(is_worker=True)
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            SVM_pb2_grpc.add_SVMServicer_to_server(worker, server)
            worker.my_server = server
            server.add_insecure_port(address)
            server.start()
        except KeyboardInterrupt:
            server.stop(0)

        return server


    def start_(self,address):
        return self.serve(address)

        #while not worker.complete:
            #time.sleep(30)
