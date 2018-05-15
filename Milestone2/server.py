import grpc
import SVM_pb2
import SVM_pb2_grpc

_ONE_DAY_IN_SECONDS = 24*3600

class SVMServicer(SVM_pb2_grpc.SVMServicer):

    def __init__(self, is_worker=True):
        '''opening both training and test files	'''
        self.file = open('data/labels_balanced.dat','r')
        self.file1 = open('data/test_labels_balanced.dat', 'r')
        self.is_worker = is_worker
        self.params = {}
        self.data = []  #each entry is a data point represented as a dict
        self.target = []
        self.val_data = {} #each entry is a validation data point represented as a dict
        self.val_target = []
        self.batch_size = None
        self.lr = 0.001
        self.rcv_grads = {}
        self.num_rcv_grads = 0
        self.coordinator_address = None
        self.workers_address = None
        self.coordinator_stub = None
        self.worker_stubs = [] #contains information about all other workers, their ports, etc.
        self.reg = 0
        self.du = {}
        self.num_nodes_update = 0
        self.losses = []
        self.tot_iter = None
        self.complete = 0
        

    def vec_mul(self, vec1, vec2):
        '''dot product of two sparse vectors	'''
        result = 0

        for elem in vec2:

            result += elem[1] * vec1.get(elem[0], 0)
			
        return result

    def ele_vec_mul(self, vec1, vec2):
        '''element wise product of two sparse vectors	'''
        result = {}

        for elem in vec2:

            result[elem] = vec2[elem] * vec1.get(elem, 0)
			
        return result

    def add_to(self, vec, sub_vec):
        '''sum of two sparse vectors	'''
        
        result = {}
        
        for key, val in sub_vec.items():

            result[key] = vec.get(key, 0) + val

        return result

    def weight_msg_to_dict(self, msg):
        '''converts the received proto message into a dictionary	'''

        ret = {}

        for entry in msg.entries:

            ret[entry.index] = entry.value

        return ret

    def prediction(self, label, pred):
        '''performs predictions	'''

        ok = (pred >= 0 and label == 1) or (pred < 0 and label == -1)

        return 1 if ok else 0

    def scalar_vec_mul(self, scalar, vec):
        '''computes multiplication of a scalar with a vector	'''

        return dict([(entry, vec[entry] * scalar) for entry in vec])
    
    def calc_reg(self, sample):
        
        regularizer = {}
        
        for key in sample:
            regularizer[key] = 2*self.params.get(key, 0)*self.du.get(key, 0)
            
        return regularizer

    def compute_gradient(self, random_indices):
        '''compute gradient for a given training examples	'''
        
        batch_grad = {}

        for i in random_indices:
            tmp = self.vec_mul(self.params, self.data[i])
            #pred = self.prediction(self.target[i], tmp)
            tmp = tmp*self.target[i]
    
            if (tmp < 1):    
                grad = self.scalar_vec_mul(-self.target[i], self.data[i])  
            else:    
                grad = dict()
                
            grad = self.add_to(grad, self.calc_reg(self.data[i]))
                
            batch_grad = self.add_to(batch_grad, grad)
    
        return batch_grad
        #return grad, tmp, pred

    def dict_to_weight_msg(self, dic, label):
        '''converts a dictionary into a proto message	'''

        ret = SVM_pb2.Row(label = label)
        entries = []

        for key, value in dic.items():

            entries.append(SVM_pb2.Entry(index = key, value = value))

        ret.entries.extend(entries)

        return ret
    
    def SendCompletionSignal(self, request, context):
        '''send/get information about learning configuration'''

        self.complete = 1    
        
        return SVM_pb2.Null()
    
    def SendLearningInfo(self, request, context):
        '''send/get information about learning configuration'''

        self.lr = request.lr
        self.batch_size = request.batch_size
        self.reg = request.reg
        self.tot_iter = request.tot_iter   
        self.start = True
        
        return SVM_pb2.Null()
    
    def SendNodeInfo(self, request, context):
        '''send/get information about ports and ips of other workers and co-ordinator '''
        
        self.coordinator_address = request.coordinator_address
        self.coordinator_stub = SVM_pb2_grpc.SVMStub(grpc.insecure_channel(self.coordinator_address))
        
        self.workers_address = request.workers_address
        
        for worker in self.workers_address:
            self.worker_stubs.append(SVM_pb2_grpc.SVMStub(grpc.insecure_channel(worker)))
        
        return SVM_pb2.Null()
    
    def UpdateSignal(self, request, context):
        '''receives weight indices from other workers, accumulates them and maintains a counter for # of updates received '''

        self.num_nodes_update += 1
        
        return SVM_pb2.Null()
	
    def GetWeights(self, weightUpdate, context):
        '''receives weight indices from other workers, accumulates them and maintains a counter for # of updates received '''
        #self.rcv_grads = {}self.num_rcv_grads = 0
        weightUpdate = self.weight_msg_to_dict(weightUpdate)
        self.rcv_grads = self.add_to(self.rcv_grads, weightUpdate)
        self.num_rcv_grads += 1
        
        if ~self.is_worker:
            self.update_weight(weightUpdate)
            loss, acc = self.compute_loss_acc(self.data, self.target)
            self.losses.append[loss]
            self.acc.append[acc]
        
        return SVM_pb2.Null()
        
        '''
		label = weightUpdate.label
		indexes = weightUpdate.indexes
		examples, labels = self.load_data(indexes, label)
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
        

		return self.dict_to_weight_msg(grad)'''
        
    

    def load_data(self, indexes, label):
        '''load the relevant examples into main memory using the seek positions sent by the client	'''
        batch_examples = []
        batch_labels = []

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


    def update_weight(self, grad):
        
        self.params = self.add_to(self.params, self.scalar_vec_mul(-self.lr, grad))   
        
        return
    
    def compute_loss_acc(self, data, target):
        
        loss = 0
        acc = 0
        
        for d in data:
            tmp = self.vec_mul(self.params, d)
            loss += max(0, 1 - tmp)
            
            if tmp*target >= 0:
                acc +=1
                
        acc /= len(target)
        
        return loss, acc