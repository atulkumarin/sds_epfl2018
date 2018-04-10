from concurrent import futures
import time

import grpc
import sys
import SVM_pb2
import SVM_pb2_grpc
from threading import Thread

_ONE_DAY_IN_SECONDS = 24*3600

i = 0


class SVMServicer(SVM_pb2_grpc.SVMServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        self.data = []
        self.labels = []

    def vec_mul(self, vec1, vec2):
        result = 0
        
        for elem in vec2.entries:
            result += elem.value*vec1.get(elem.index,0)
            
        return result

    def add_to(self, vec,sub_vec):
        # In place addition
        for key,val in sub_vec.items():
            vec[key] = vec.get(key,0) + val
        return



    def weight_msg_to_dict(self, msg):
        ret = {}
        for entry in msg.entries:
            ret[entry.index] = entry.value
        return ret



    def scalar_vec_mul(self, scalar, vec):
        return dict([(entry.index, entry.value*scalar)for entry in vec.entries])


    def compute_gradient(self, weights, index, nb_pts):
        data_sample = self.data[index]
        target = self.labels[index]
        # Intermediary compute step that will be used for computing loss
        tmp = target*self.vec_mul(weights, data_sample)
        print("Data = {}".format([(entry.index, entry.value)for entry in data_sample.entries]))
        if (tmp < 1):
            grad = self.scalar_vec_mul(-target, data_sample)
        else:
            grad = dict()
        return grad, tmp


    def dict_to_weight_msg(self, dic):
        ret = SVM_pb2.Row(label = 'weight')
        entries = []
        for key, value in dic.items():
             entries.append(SVM_pb2.Entry(index = key,value = value))
        ret.entries.extend(entries)  #iter??
        return ret
    
    def GetWeights(self, weightUpdate, context):
        global i
        i += 1
        print("Iteration #{} ".format(i), end=" ")
        # Function used for Getting the weight vector and the batch index to use for training
        indexes = weightUpdate.indexes
        #print("Indices : {}".format(indexes))
        weights = self.weight_msg_to_dict(weightUpdate.row)
        #print('received {} and {}'.format(indexes.label,weights.label))
        #return SVM_pb2.Row(label = 'OK')
        grad = {}
        loss = 0
        nb_pts = len(indexes)
        for index in indexes:
            sub_grad,sub_loss = self.compute_gradient(weights, index, nb_pts)
            self.add_to(grad,sub_grad)
            loss += max(0,1-sub_loss)
        grad[-1] = loss
        
        #print("Grad={}".format(grad))
        return self.dict_to_weight_msg(grad)

    def GetData(self, matrix, context):
        # Used to fetch data from coordinator
        self.data = matrix.rows
        self.labels = matrix.categories
        return SVM_pb2.Status(status = 'OK')


def serve(port):
    # Creates worker
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
    # Start nb_workers workers in parallel
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
