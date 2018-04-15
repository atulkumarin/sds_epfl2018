from concurrent import futures
import time
import grpc
import sys
import SVM_pb2
import SVM_pb2_grpc
from threading import Thread

_ONE_DAY_IN_SECONDS = 24*3600

class SVMServicer(SVM_pb2_grpc.SVMServicer):

    def __init__(self):

        self.file = open('data/labels_balanced.dat','r')
        self.file1 = open('data/test_labels_balanced.dat', 'r')

    def vec_mul(self, vec1, vec2):

        result = 0

        for elem in vec2:

            result += elem[1] * vec1.get(elem[0], 0)
            
        return result

    def add_to(self, vec, sub_vec):

        for key, val in sub_vec.items():

            vec[key] = vec.get(key, 0) + val

        return

    def weight_msg_to_dict(self, msg):

        ret = {}

        for entry in msg.entries:

            ret[entry.index] = entry.value

        return ret

    def prediction(self, label, pred):

        ok = (pred >= 0 and label == 1) or (pred < 0 and label == -1)

        return 1 if ok else 0

    def scalar_vec_mul(self, scalar, vec):

        return dict([(entry[0], entry[1] * scalar) for entry in vec])

    def compute_gradient(self, weights, example,label):

        tmp = self.vec_mul(weights, example)
        pred = self.prediction(label, tmp)
        tmp = tmp * label

        if (tmp < 1):

            grad = self.scalar_vec_mul(-label, example)

        else:

            grad = dict()

        return grad, tmp, pred

    def dict_to_weight_msg(self, dic):

        ret = SVM_pb2.Row(label = 'weight')
        entries = []

        for key, value in dic.items():

             entries.append(SVM_pb2.Entry(index = key, value = value))

        ret.entries.extend(entries)  #iter??
        return ret
    
    def GetWeights(self, weightUpdate, context):
        
        label = weightUpdate.label
        indexes = weightUpdate.indexes
        examples,labels = self.load_data(indexes, label)
        weights = self.weight_msg_to_dict(weightUpdate.row)

        grad = {}
        loss = 0
        accuracy = 0
        nb_pts = len(indexes)

        for i in range(len(indexes)):

            if label == 'train':

                sub_grad, sub_loss, pred = self.compute_gradient(weights, examples[i], labels[i])
                accuracy += pred
                self.add_to(grad,sub_grad)
                loss += max(0, 1 - sub_loss)

            else:

                _, sub_loss, pred = self.compute_gradient(weights, examples[i], labels[i])
                accuracy += pred
                loss += max(0, 1 - sub_loss)

        accuracy = float(accuracy)/nb_pts
        grad[-1] = loss/nb_pts
        grad[-2] = accuracy

        return self.dict_to_weight_msg(grad)

    def load_data(self, indexes, label):

        batch_examples = []
        batch_labels =[]

        for idx in indexes :

            if label == 'train':

                self.file.seek(idx)
                sample_ = self.file.readline()

            else:

                self.file1.seek(idx)
                sample_ = self.file1.readline()

            sample = sample_.split(' ')
            entries = []

            for i in range(2, len(sample) - 1):

                entry = sample[i].split(':')
                entries.append((int(entry[0]), float(entry[1])))

            batch_examples.append(entries)

            if sample[-1].strip() == '':

                print(idx)

            batch_labels.append(int(sample[-1].strip()))
            
        return batch_examples, batch_labels

def serve(port):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
    SVM_pb2_grpc.add_SVMServicer_to_server(SVMServicer(), server)
    server.add_insecure_port('[::]:' + str(port))
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
        port += 1

    for thread in threads:
        thread.join()
