#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comments and Info to be added
"""

from concurrent import futures
import time
import grpc
import sgd_svm_pb2_grpc
import sgd_svm_pb2

import numpy as np
import random

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

data_samples = (np.random.randn(50,2)*2 + 5).tolist() + (np.random.randn(50,2)*2 - 5).tolist()
data_samples = [str(0) + ':' + str(i[0]) + ' ' + str(1) + ':' + str(i[1]) for i in data_samples]
params = data_samples
targets = [1]*50 + [-1]*50


class SGD_SVMServicer(sgd_svm_pb2_grpc.SGD_SVMServicer):
    """Provides methods that implement functionality of sgd_svm server."""
    
    def GetData(self, request, context):
            n = random.randint(0, len(data_samples) - 1)
            return sgd_svm_pb2.DataSample(data_sample=data_samples[n], param=params[n], target=str(targets[n]))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sgd_svm_pb2_grpc.add_SGD_SVMServicer_to_server(SGD_SVMServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()