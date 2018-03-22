from concurrent import futures
import time
import math

import grpc
import sys
import SVM_pb2
import SVM_pb2_grpc
from threading import Thread

_ONE_DAY_IN_SECONDS = 24*3600

def convert_msg_to_matrix(msg):
    matrix = []
    for row in msg.rows:
        row = []
        for value in row.values:
            row.append(value)
        matrix.append(row)
    return matrix



class SVMServicer(SVM_pb2_grpc.SVMServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        self.data = []

    def GetWeights(self, request_iterator, context):
        indexes = request_iterator.next()
        weights = request_iterator.next()
        print('received {} and {}'.format(indexes.label,weights.label))
        return SVM_pb2.Row(label = 'OK')



    def GetData(self, request_iterator, context):
        for matrix in request_iterator:
            self.data.append(matrix)

        return SVM_pb2.Status(status = 'OK')


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    SVM_pb2_grpc.add_SVMServicer_to_server(
        SVMServicer(), server)
    server.add_insecure_port('[::]:'+str(port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    nb_workers = int(sys.argv[1])
    threads = []
    port = 50051
    for i in range(nb_workers):
        thread = Thread(target = serve, args = (port, ))
        thread.start()
        threads.append(thread)
        port +=1
    for thread in threads:
        thread.join()
