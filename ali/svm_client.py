
import grpc
import SVM_pb2
import SVM_pb2_grpc
import sys 
import numpy as np
from threading import Thread

channels =[]


nb_batches = 0
matrix =None
responses = None
lr = None
reg_factor = None
weights = {}
def load_data(path_features,path_label):
    # Open files
    features  = open(path_features,'r')
    labels = open(path_label,'r')

    topics = []
    # Reads all samples
    lines = features.readlines()
    nb_data_pt = len(lines)

    # Fetch id of positive samples
    lines_labels = [ int(i) for i in labels.readlines()]
    labels.close()
    features.close()


    nb_batches = int(len(lines)/batch_size)
    list_data =[]
    labels =[]
    
  

    # Build the dataset
    for index,line in enumerate(lines):
        # Fetch example id + features
        splitted_line = line.split(' ')
        id_line = splitted_line[0]
        msg_row = SVM_pb2.Row(label = id_line)
        entries = []
        # Fetch for every non zero feature its index and value
        for i in range(2,len(splitted_line)):
            entry = splitted_line[i].split(':')
            # Append feature to row entries
            msg_row.append(SVM_pb2.Entry(index = int(entry[0]),value = float(entry[1])))
        msg_row.entries.extend(entries)

        # Add row to data matrix
        list_data[i].append(msg_row)
        labels[i].append(lines_labels[i])
    # Create batches message objects
    matrix = SVM_pb2.Matrix(label = 'data')
    matrix.rows.extend(list_data)
    matrix.categories.extend(labels)

    return matrix,nb_data_pt







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
    
    
def mat_mul(mat1, mat2):
    """mat1 is (_n x _m)
       mat2 is (_m x _p )
       resulting matrix will be (_n x _p)
    """
    _n = len(mat1)
    _m = len(mat1[0])
    _p = len(mat2[0])
    
    result = [[0 for x in range(_p)] for y in range(_n)] 
    
    for i in range(_n):
        for j in range(_p):
            for k in range(_m):
                result[i][j] += mat1[i][k]*mat2[k][j]
        
    return result
    

def compute_gradient(param, data_sample, target, lrate=0.2):
    
    if (target*vec_mul(param, data_sample) < 1):
        grad = scalar_vec_mul(-1*target, data_sample)
    else:
        grad = [0 for x in range(len(param))]
    
    grad = vec_sum(grad, scalar_vec_mul(2*lrate, param))

    return grad
    
    

def send_data(stub,i):
    global responses
    responses[i] = stub.GetData(iter(matrix))
    return
def send_weights(stub,i,data):
    global responses
    responses[i] = stub.GetWeights(data)
    return    

def run():
    global channels,matrices,responses,lr,reg_factor
    matrix,nb_data_pt = load_data('../data/lyrl2004_vectors_test_pt0.dat','../data/labels.txt')
    print('Data loaded')
    
    stubs = []
    for channel in channels :
        stub = SVM_pb2_grpc.SVMStub(grpc.insecure_channel('localhost:{}'.format(channel)))
        stubs.append(stub)

    threads = []
    # Send data
    for i,stub in enumerate(stubs):
        thread = Thread(target = send_data, args = (stub,i,))
        thread.start()
        threads.append(thread)
        
    [th.join() for th in threads]

    for index,response in enumerate(responses):
        if response.status == 'OK':
            print('Worker {} received data'.format(index))
        else :
            print('Worker {} did not receive data'.format(index))
    del matrix
    data_order = list(np.permutation(nb_data_pt))
    n_epoch = 20
    epoch= 0
    i = 0
    # Training loop
    while(epoch < n_epoch):

        entries =[] 
        # create weight msg
        for  key, value in weights.items():
            entries.append(SVM_pb2.Entry(index = key ,value = value))
        msg_row = SVM_pb2.Row()
        msg_row.extend(entries)


        threads = []

        # send weights along with data points indexes

        for j,stub in enumerate(stubs):
            st = i
            end = i+batch_size
            msg = SVM_pb2.WeightUpdate(row = msg_row ,indexes = iter(data_order[st:end]))
            thread = Thread(target = send_weights, args = (stub,j,msg,))
            thread.start()

            if i > nb_data_pt:
                i= 0 
                epoch +=1
                data_order = list(np.permutation(nb_data_pt))

            else:
                i+= batch_size

            threads.append(thread)
        join_threads(threads)
        loss = update_weight(nb_data_pt,batch_size,lr,reg_factor)


def update_weight(nb_data_pt,batch_size,lr,reg_factor):
    global weights
    gradient = {}
    loss = 0
    normalizer = len(responses)
    # TODO : SHOULD WE NORMALIZE BY NB OF WORKERS ?
    for response in responses:
        for entry in response.entries:
            # Loss is stored in the gradient vector with index -1
            if entry.index == -1:
                loss += entry.value
            else:
                gradient[entry.index] = gradient.get(index,0) + value
    for k,v in weights.items():
        gradient[k] += gradient.get(k,0)+v*reg_factor
        loss += (v**2)*reg_factor
    for key,value in gradient:
        # Update weight vector and add regularization
        weights[key] = weights.get(key,0) + lr*(value)

    return loss

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
    lr = 0.001
    reg_factor = 0.002
    responses = [None]*nb_workers
    for i in range(nb_workers):
        channels.append(50051+i)
    run()
