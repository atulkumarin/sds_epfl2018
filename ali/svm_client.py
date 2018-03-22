
import grpc
import SVM_pb2
import SVM_pb2_grpc
import sys 
import numpy as np
from threading import Thread

channels =[]

nb_batches = 0
matrices =None
responses = None
def load_data(path_features,path_label,batch_size):
    global nb_batches
    features  = open(path_features,'r')
    labels = open(path_label,'r')

    topics = []
    lines = features.readlines()
    lines_labels = set([ int(i) for i in labels.readlines()])
    labels.close()
    features.close()

    nb_batches = int(len(lines)/batch_size)
    list_data =[]
    labels =[]
    for i in range(nb_batches):
        labels.append([])
        list_data.append([])
        
    order = np.floor(np.random.permutation(len(lines))/batch_size)
    for index,line in enumerate(lines):
        splitted_line = line.split(' ')
        id_line = splitted_line[0]
        msg_row = SVM_pb2.Row(label = id_line)
        entries = []
        for i in range(2,len(splitted_line)):
            entry = splitted_line[i].split(':')
            entries.append(SVM_pb2.Entry(index = int(entry[0]),value = float(entry[1])))
        msg_row.entry.extend(entries)
        list_data[int(order[index])].append(msg_row)
        labels[int(order[index])].append(1 if int(id_line) in lines_labels else -1)
    matrices = []
    for i in range(nb_batches):
        matrix = SVM_pb2.Matrix(label = 'data')
        matrix.rows.extend(list_data[i])
        matrix.categories.extend(labels[i])
        matrices.append(matrix)
    return matrices







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
    responses[i] = stub.GetData(iter(matrices))
    return
        

def run():
    global channels,matrices,responses
    matrices = load_data('../data/lyrl2004_vectors_test_pt0.dat','../data/labels.txt',100)
    print('Data loaded')
    
    nb_batches_per_worker = int(len(matrices)/len(channels))
    stubs = []
    for channel in channels :
        stub = SVM_pb2_grpc.SVMStub(grpc.insecure_channel('localhost:{}'.format(channel)))
        stubs.append(stub)

    threads = []

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


def make_label_file(label_kept,features,labels):
    features  = open(features,'r')
    labels = open(labels,'r')
    out = open('../data/output.txt','w')
    ids = set([(i.split(' ')[0]) for i in features.readlines()])
    for label in labels.readlines():
        label = label.split(' ')
        if label[0] == label_kept and label[1] in ids:
            out.write(label[1]+'\n')
    features.close()
    labels.close()
    out.close()

if __name__ == '__main__':
    nb_workers = int(sys.argv[1])
    responses = [None]*nb_workers
    for i in range(nb_workers):
        channels.append(50051+i)
    run()
