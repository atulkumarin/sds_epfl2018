from concurrent import futures
import time

import grpc
import sys
import SVM_pb2
import SVM_pb2_grpc
from threading import Thread

_ONE_DAY_IN_SECONDS = 24*3600



class SVMServicer(SVM_pb2_grpc.SVMServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        self.file = open('data/labels_balanced.dat','r')


    def vec_mul(self, vec1, vec2):
        result = 0
        for elem in vec2:
            result += elem[1]*vec1.get(elem[0],0)
            
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

    def prediction(self,label,pred):
        ok = (pred >= 0 and label == 1) or (pred < 0 and label == -1)
        return 1 if ok else 0

    def scalar_vec_mul(self, scalar, vec):
        return dict([(entry[0], entry[1]*scalar) for entry in vec])


    def compute_gradient(self, weights, example,label):

        # Intermediary compute step that will be used for computing loss
        tmp = self.vec_mul(weights, example)
        pred = self.prediction(label,tmp)
        tmp = tmp*label
        if (tmp < 1):
            grad = self.scalar_vec_mul(-label, example)
        else:
            grad = dict()
        return grad,tmp,pred


    def dict_to_weight_msg(self, dic):
        ret = SVM_pb2.Row(label = 'weight')
        entries = []
        for key, value in dic.items():
             entries.append(SVM_pb2.Entry(index = key,value = value))
        ret.entries.extend(entries)  #iter??
        return ret
    
    def GetWeights(self, weightUpdate, context):
        # Function used for computing the weight vector

        indexes = weightUpdate.indexes
        examples,labels = self.load_data(indexes)
        weights = self.weight_msg_to_dict(weightUpdate.row)
        #print('received {} and {}'.format(indexes.label,weights.label))
        #return SVM_pb2.Row(label = 'OK')
        grad = {}
        loss = 0
        accuracy = 0
        nb_pts = len(indexes)
        for i in range(len(indexes)):
            sub_grad,sub_loss,pred = self.compute_gradient(weights, examples[i],labels[i])
            accuracy+= pred
            self.add_to(grad,sub_grad)
            loss += max(0,1-sub_loss)
        accuracy = float(accuracy)/nb_pts
        grad[-1] = loss/nb_pts
        grad[-2] = accuracy
        

        #print("Grad={}".format(grad))
        return self.dict_to_weight_msg(grad)

    def GetData(self, matrix, context):
        # Used to fetch data from coordinator
        self.data_indexes = matrix
        self.file = open('data/data_balanced.dat','r')

        return SVM_pb2.Status(status = 'OK') 
    def load_data(self,indexes):
        # Fetch data pts from file
        batch_examples = []
        batch_labels =[]
        for idx in indexes :
            self.file.seek(idx)
            sample_ = self.file.readline()
            sample = sample_.split(' ')
            entries = []
            # Fetch for every non zero feature its index and value
            for i in range(2,len(sample)-1):
                entry = sample[i].split(':')
                # Append feature to row entries
                entries.append((int(entry[0]),float(entry[1])))
            #entries.append((-3,1.0))

            batch_examples.append(entries)
            #try:
            if sample[-1].strip() == '':
                print(idx)
            batch_labels.append(int(sample[-1].strip()))
            # except ValueError:
            #     print('LINE ID : {}'.format(sample[0]))

        return batch_examples,batch_labels

def serve(port):
    # Creates worker
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
