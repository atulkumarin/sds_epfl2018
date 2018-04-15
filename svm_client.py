import grpc
import SVM_pb2
import SVM_pb2_grpc
import sys 
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
from random import shuffle

channels = []
batch_size = 80
responses = None
lr = None
reg_factor = None
weights = {}
losses = []
test_losses = []
accuracies = []
test_accuracies = []

def load_data_and_get_points(path):
    
    map_point_to_seek = [0]
    cnt = 0
    
    with open(path,'r') as features_and_labels:
        
        for line in iter(features_and_labels.readline, ''):
            cnt += 1
            map_point_to_seek.append(features_and_labels.tell())

    return map_point_to_seek[:-1], cnt
    
def send_weights(stub, i, data):

    global responses
    responses[i] = stub.GetWeights.future(data)
    return    

def collect_results():

    global responses
    
    for i in range(len(responses)):
        
	responses[i] = responses[i].result()

def run():

    global channels, responses, lr, reg_factor, batch_size, losses

    indexes, nb_data_pt = load_data_and_get_points('data/labels_balanced.dat')
    test_indexes, test_nb_data_pt = load_data_and_get_points('data/test_labels_balanced.dat')
    print('Seek positions loaded.')
    
    stubs = []
    for channel in channels :
        stub = SVM_pb2_grpc.SVMStub(grpc.insecure_channel('localhost:{}'.format(channel)))
        stubs.append(stub)

    threads = []

    epoch = 0
    i = 0
    x = 0
    
    data_indexes = indexes
    shuffle(indexes)

    while(True):

        entries = [] 
        
        for key, value in weights.items():
            entries.append(SVM_pb2.Entry(index = key, value = value))
            
        msg_row = SVM_pb2.Row()
        msg_row.entries.extend(entries)

        #threads = []

        k = 0
        flag = 0

        for j,stub in enumerate(stubs):

            if x % 10 == 0 and x != 0:
                
                if k > test_nb_data_pt:
                    epoch += 1
                    break

                st = k
                end = min(k + int(test_nb_data_pt/len(stubs)), test_nb_data_pt - 1)
                msg = SVM_pb2.WeightUpdate(row = msg_row, indexes = test_indexes[st:end], label = 'test')
                future = send_weights(stub, j, msg)
                k += int(test_nb_data_pt/len(stubs))
                flag = 1

            else:
		
                flag = 0
                if i > nb_data_pt:
                    i = 0 
                    epoch += 1
                    
                st = i
                end = min(i + batch_size, nb_data_pt - 1)
                msg = SVM_pb2.WeightUpdate(row = msg_row, indexes = data_indexes[st:end], label = 'train')
                #thread = Thread(target = send_weights, args = (stub,j,msg,))
                future = send_weights(stub, j, msg)
                #thread.start()
                i += batch_size

            #threads.append(thread)
            
        #join_threads(threads)

        collect_results()
        print('Current iteration : {}'.format(x))
        print('Current epoch     : {}'.format(epoch))
        x += 1

        if flag == 1:

            test_loss, test_acc = test_metric()
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            plt.plot(list(range(len(test_losses))), test_losses)
            plt.savefig("test_loss.png")
            plt.close()
            plt.plot(list(range(len(test_accuracies))), test_accuracies)
            plt.savefig("test_acc.png")
            plt.close()
            print('Plot Saved')
            print("Test loss = {}\n".format(test_loss))

        else:

            loss, acc = update_weight(nb_data_pt,batch_size,lr,reg_factor)
            losses.append(loss)
            accuracies.append(acc)

            if(x % 20 == 0):

                plt.plot(list(range(len(losses))), losses)
                plt.savefig("train_loss.png")
                plt.close()
                plt.plot(list(range(len(accuracies))), accuracies)
                plt.savefig("train_acc.png")
                plt.close()
                print('Plot Saved')

            print("Loss = {}\n".format(loss))
            #print("Weights = {}".format(weights))
             
def test_metric():

    global weights
    loss = 0
    accuracy = 0

    for response in responses:

        for entry in response.entries:

            if entry.index == -1:

                loss += entry.value

            if entry.index == -2:

                accuracy += entry.value

    accuracy = accuracy/len(responses)
    loss = loss/len(responses)

    return loss, accuracy

def update_weight(nb_data_pt, batch_size, lr, reg_factor):

    global weights
    gradient = {}
    loss = 0
    accuracy = 0

    for response in responses:

        for entry in response.entries:
            
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

        weights[key] = weights.get(key,0) - lr*(value)

    return loss, accuracy


def join_threads(threads):

    [th.join() for th in threads]
    return

if __name__ == '__main__':
    nb_workers = int(sys.argv[1])
    lr = 0.1
    reg_factor = 0
    responses = [None] * nb_workers
    for i in range(nb_workers):
        channels.append(50051 + i)
    run()
