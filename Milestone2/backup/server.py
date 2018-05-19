import grpc
import SVM_pb2
import SVM_pb2_grpc
import configuration as config
from threading import Thread, Lock
_ONE_DAY_IN_SECONDS = 24 * 3600
from utils import *
from concurrent import futures


class SVMServicer(SVM_pb2_grpc.SVMServicer):

    def __init__(self, config_file='config.json', is_worker=True):
        '''opening both training and test files '''
        self.params = {}
        self.val_data = {}  # each entry is a validation data point represented as a dict
        self.val_target = []
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.rcv_grads = {}
        self.coordinator_address = None
        self.workers_address = None
        self.coordinator_stub = None
        self.worker_nb = -1
        # contains information about all other workers, their ports, etc.
        self.worker_stubs = []
        self.reg = config.reg
        self.tot_iter = config.tot_iter
        self.is_worker = is_worker
        self.val_data_size = config.val_data_size
        self.log_file = None
        self.log_file_lock = Lock()
        self.test_after = config.test_after
        self.currently_computed_elem = -1
        self.currently_computed_elem_lock = Lock()
        if is_worker:
            # each entry is a data point represented as a dict
            self.data, self.target, self.du = load_data(config.train_data_path)
            print('[INFO] LOADED WORKER DATA')
        else:
            self.data, self.target, self.du = load_data(
                config.test_data_path, nb_sample=config.val_data_size)
            self.log_file = open(config.log_file_path, 'w')
            print('[INFO] LOADED COORD DATA')

        self.my_server = None
        self.complete = False

    def SendCompletionSignal(self, request, context):
        '''send/get information about learning configuration'''

        if not self.is_worker:
            for worker in self.worker_stubs:
                worker.SendCompletionSignal(SVM_pb2.Null())

        self.complete = True
        self.my_server.stop(0)

        return SVM_pb2.Null()

    def SendLearningInfo(self, request, context):
        '''send/get information about learning configuration'''

        self.lr = request.lr
        self.batch_size = request.batch_size
        self.reg = request.reg
        self.tot_iter = request.tot_iter

        self.start = True

        return SVM_pb2.Null()

    def Start(self, request, context):
        if self.is_worker:
            print('[INFO] Worker {} started computing'.format(self.worker_nb))
            self.start_computation_worker_asynch()

    def SendNodeInfo(self, request, context):
        '''send/get information about ports and ips of other workers and co-ordinator '''

        self.coordinator_address = request.coordinator_address
        self.coordinator_stub = SVM_pb2_grpc.SVMStub(
            grpc.insecure_channel(self.coordinator_address))

        self.workers_address = request.workers_address

        for worker in self.workers_address:
            self.worker_stubs.append(
                SVM_pb2_grpc.SVMStub(grpc.insecure_channel(worker)))

        self.worker_nb = request.worker_nb
        if self.is_worker:
            print('[INFO] WORKER {} received all addresses'.format(self.worker_nb))
        return SVM_pb2.Null()

    def GetWeights(self, weightUpdate, context):
        '''receives weight indices from other workers, accumulates them and maintains a counter for # of updates received '''

        worker_nb = weightUpdate.worker_nb
        iter_num = weightUpdate.iteration_number
        weightUpdate = weight_msg_to_dict(weightUpdate)

        if not self.is_worker:
            with self.log_file_lock:
                self.log_file.write('[worker {}] it: {} Loss: {} acc: {}\n'.format(
                    worker_nb, iter_num, weightUpdate[-1], weightUpdate[-2]))
                print('[INFO] Coordinator log: [worker {}] it: {} Loss: {} acc: {} '.format(
                    worker_nb, iter_num, weightUpdate[-1], weightUpdate[-2]))
            del weightUpdate[-1]
            del weightUpdate[-2]
            self.update_weight(weightUpdate)
            cond = False
            with self.currently_computed_elem_lock:
                thresh_test = iter_num - self.currently_computed_elem
                if thresh_test >= self.test_after:
                    self.currently_computed_elem = iter_num
                    cond = True
            if cond:
                loss, acc = self.compute_test()

                with self.log_file_lock:
                    self.log_file.write(
                        '[TEST] it: {} Loss: {} acc: {} \n'.format(iter_num, loss, acc))
                    print('[INFO] TEST it: {} Loss: {} acc: {} \n'.format(
                        iter_num, loss, acc))

        else:
            add_to(self.rcv_grads, weightUpdate,
                   res_vector=self.rcv_grads)
            #print('[INFO] Worker {} received vector from worker {}'.format(self.worker_nb,worker_nb))

        return SVM_pb2.Null()

    def update_weight(self, grad):
        add_to(self.params, scalar_vec_mul(-self.lr, grad),
               res_vector=self.params)
        return

    def start_computation_worker_asynch(self):
        random_indices = random.sample(
            range(len(self.data)), len(self.data))
        print('[INFO] indices train {}'.format(len(random_indices)))

        iter_num = 0
        start, end = 0, self.batch_size
        epoch = 0
        while iter_num < self.tot_iter:  # change for stopping criteria
            # training start
            if start >= len(self.data):
                start, end = 0, self.batch_size
                epoch += 1

            grad, acc, loss = compute_gradient(
                self, random_indices[start:end])
            grad_msg = dict_to_weight_msg(grad)
            grad_msg.iteration_number = iter_num
            grad_msg.worker_nb = self.worker_nb
            for stub in self.worker_stubs:
                stub.GetWeights.future(grad_msg)

            grad_msg.entries.extend([SVM_pb2.Entry(index=-1, value=loss)])
            grad_msg.entries.extend([SVM_pb2.Entry(index=-2, value=acc)])

            self.coordinator_stub.GetWeights.future(grad_msg)

            total_grad = add_to(grad, self.rcv_grads)

            self.update_weight(total_grad)

            # MAYBE LOCK
            self.rcv_grads = {}

            iter_num += 1

            start += self.batch_size
            end += self.batch_size

        self.coordinator_stub.SendCompletionSignal(SVM_pb2.Null())

    def compute_test(self):
        print('[INFO] Started computing validation Loss and acc')
        return compute_loss_acc(self, self.data, self.target)
