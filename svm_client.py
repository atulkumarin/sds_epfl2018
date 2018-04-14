
import grpc
import SVM_pb2
import SVM_pb2_grpc
import sys 
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
from random import shuffle

channels =[]


nb_batches = 0
batch_size = 1
matrix =None
responses = None
lr = None
reg_factor = None
weights = {}
losses =[]
accuracies =[]

def test_acc():
    #TODO : TEST
        global test_data
        for  key, value in weights.items():
            entries.append(SVM_pb2.Entry(index = key ,value = value))
            
        msg_row = SVM_pb2.Row()
        msg_row.entries.extend(entries)


        for j,stub in enumerate(stubs):

            if i > nb_data_pt:
                i= 0 
                epoch +=1
                #data_order = shuffle(data_order)
            st = i
            end = min(i+batch_size,nb_data_pt-1)
            msg = SVM_pb2.WeightUpdate(row = msg_row ,indexes = data_indexes[st:end])
            #thread = Thread(target = send_weights, args = (stub,j,msg,))
            future = send_weights(stub,j,msg)
            #thread.start()
            i+= batch_size

        collect_results()

def load_data_and_get_points(path):
    # Open files 
    #labels = open(path_label,'r')
    map_point_to_seek = [0]
    cnt = 0
    with open(path,'r') as features_and_labels:
        for line in iter(features_and_labels.readline, ''):
                cnt+=1
                map_point_to_seek.append(features_and_labels.tell())

    return map_point_to_seek[:-1],cnt

def scalar_vec_mul(scalar, vec):
    return [scalar*i for i in vec]

def scalar_vec_sum(scalar, vec):
    return [scalar + i for i in vec]

def vec_sum(vec1, vec2):
    return [vec1[i] + vec2[i] for i in range(len(vec1))]

def vec_mul(vec1, vec2):
    num_elements = len(vec1)

    result = 0
    
    for i in range(num_elements):
        result += vec1[i]*vec2[i]
        
    return result
    

def send_data(stub,i,data):
    global responses
    responses[i] = stub.GetData(data)
    return

def send_weights(stub,i,data):
    global responses
    responses[i] = stub.GetWeights.future(data)
    return    
def collect_results():
    global responses
    for i in range(len(responses)):
        responses[i] = responses[i].result()

def run():
    global channels, matrix, responses, lr, reg_factor, batch_size, losses
    #matrix, nb_data_pt = load_data('./data/train_1000.dat','./data/labels_1000.txt')
    indexes, nb_data_pt = load_data_and_get_points('data/labels_balanced.dat')
    print('seek positions loaded')
    # connect to stubs and Send seek positions
    stubs = []
    for channel in channels :
        stub = SVM_pb2_grpc.SVMStub(grpc.insecure_channel('localhost:{}'.format(channel)))
        stubs.append(stub)

    threads = []


    n_epoch = 20
    epoch = 0
    i = 0
    
    data_indexes = indexes
    shuffle(indexes)

    # Training loop
    x = 0
    #while(epoch < n_epoch):
    while(True):

        entries =[] 
        # create weight msg
        for  key, value in weights.items():
            entries.append(SVM_pb2.Entry(index = key ,value = value))
            
        msg_row = SVM_pb2.Row()
        msg_row.entries.extend(entries)


        #threads = []

        # send weights along with data points indexes

        for j,stub in enumerate(stubs):

            if i > nb_data_pt:
                i= 0 
                epoch +=1
                #data_order = shuffle(data_order)
            st = i
            end = min(i+batch_size,nb_data_pt-1)
            msg = SVM_pb2.WeightUpdate(row = msg_row ,indexes = data_indexes[st:end])
            #thread = Thread(target = send_weights, args = (stub,j,msg,))
            future = send_weights(stub,j,msg)
            #thread.start()
            i+= batch_size

            #threads.append(thread)
            
        #join_threads(threads)
        collect_results()
        print('it #{}'.format(x))
        x+=1

        loss,acc = update_weight(nb_data_pt,batch_size,lr,reg_factor)
        losses.append(loss)
        accuracies.append(acc)
        if(x%20 == 0):
            plt.plot(list(range(len(losses))), losses)
            plt.savefig("train_loss.png")
            plt.close()
            plt.plot(list(range(len(accuracies))), accuracies)
            plt.savefig("train_acc.png")
            plt.close()
            print('Plot Saved')



        print("Loss = {}".format(loss))
        #print("Weights = {}".format(weights))
        
        


def update_weight(nb_data_pt,batch_size,lr,reg_factor):
    global weights
    gradient = {}
    loss = 0
    accuracy = 0
    for response in responses:
        for entry in response.entries:
            # Loss is stored in the gradient vector with index -1
            if entry.index == -1:
                loss += entry.value
            elif entry.index == -2:
                accuracy += entry.value
            else:
                gradient[entry.index] = gradient.get(entry.index,0) + entry.value
    accuracy = accuracy/len(responses)
    loss = loss/len(responses)
    for k, v in weights.items():
        gradient[k] = gradient.get(k,0) + v*reg_factor
        loss += (v**2)*reg_factor


    for key,value in gradient.items():
        # Update weight vector and add regularization
        weights[key] = weights.get(key,0) - lr*(value)

    return loss,accuracy

def join_threads(threads):
    [th.join() for th in threads]
    return


def make_label_file(label_kept,features,labels):
    features  = open(features,'r')
    labels = open(labels,'r')
    out = open('../data/labels.txt','w')
    ids = [(i.split(' ')[0]) for i in features.readlines()]
    labels_lines = dict([(x[0],x[1]) for x in labels.readlines().split(' ')])
    for id_ in ids:
        line_label = labels_lines[id_]
        if line_label in label_kept :
            out.write('1\n')
        else:
            out.write('-1\n')
    features.close()
    labels.close()
    out.close()

if __name__ == '__main__':
    nb_workers = int(sys.argv[1])
    lr = 0.1
    reg_factor = 0
    responses = [None]*nb_workers
    for i in range(nb_workers):
        channels.append(50051+i)
    run()
