import grpc
import SVM_pb2
import SVM_pb2_grpc
import configuration as config
from threading import Lock, Thread
from utils import *
import socket
import os
import time
import _thread
from concurrent import futures
import json

_ONE_DAY_IN_SECONDS = 24 * 3600

#import conf_no_docker as os


class SVMServicer(SVM_pb2_grpc.SVMServicer):

    def __init__(self, config_file='config.json', is_worker=True):
        '''Initialize worker or coordinators variable and open training or validation files and loads them to memory'''
        self.params = {}
        self.val_target = []
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.rcv_grads = {}
        self.coordinator_address = None
        self.workers_address = []
        self.coordinator_stub = None
        self.worker_nb = -1
        self.start_time = 0
        self.worker_stubs = []
        self.reg = config.reg
        self.tot_iter = config.tot_iter
        self.is_worker = is_worker
        self.val_data_size = config.val_data_size
        self.log_file = None
        self.log_file_lock = Lock()
        self.lock_end = Lock()
        self.ended_training = False
        self.test_after = config.test_after
        self.currently_computed_elem = -1
        self.currently_computed_elem_lock = Lock()
        self.current_thread = None
        self._asynch = True if (os.environ['ASNYCH'] == '1') else False
        self.iter_num = 0
        self.time_flush = time.time()
        self.my_server = None
        self.complete = False
        self.start = 0
        self.pool_threads = futures.ThreadPoolExecutor(100)
        # Check if current servicer is built for a worker
        if is_worker:
            # each entry  data point is represented as a list of tupples (component id,value)
            # Load training data
            self.data, self.target, self.du, _, _ = load_data_real(
                os.environ['TRAIN_DATA_PATH'], os.environ['LABEL_DATA_PATH'])
            print('[INFO] LOADED WORKER DATA')

        else:

            # Load validation data if the servicer is built for a coordinator
            self.data, self.target, _, self.base_idx, self.targets_all = load_data_real(
                os.environ['TEST_DATA_PATH'], os.environ['LABEL_DATA_PATH'],
                nb_sample_per_class=config.val_data_size, proba_sample=config.proba_sample, compute_du=False)
            self.log_file = open(config.log_file_path, 'w',
                                 buffering=int(0.4 * (1024**2)))
            # Use du's computed on training set
            self.du = compute_du(os.environ['TRAIN_DATA_PATH'])
            #_thread.start_new_thread(self.flush_log,())
            print('[INFO] LOADED COORD DATA')

        # Shuffle loaded data
        pairs = list(zip(self.data, self.target))
        random.shuffle(pairs)
        self.data, self.target = zip(*pairs)
        print('[INFO] DATA SIZE: {}'.format(len(self.data)))

    def compute_test(self):
        '''Function that computes the metrics over the 4 test datasets'''
        # Check if someone else ogave started computing the test accuracy
        with self.lock_end:
            if self.ended_training:
                return
            else:
                self.ended_training = True
        # Make a copy of the weight vector
        weights = self.params.copy()
        time_elapsed = time.time() - self.start_time
        log_string = '[INFO] time:{} Started saving weight vector\n'.format(
            time_elapsed)
        print(log_string, end='')
        self.log_file.write(log_string)
        # Save weight vectors
        with open(os.environ['LOG_FOLDER'] + '/weights.json', 'w') as file:
            json.dump(weights, file)
        time_elapsed = time.time() - self.start_time
        log_string = '[INFO] time:{} Saved weight vector\n'.format(
            time_elapsed)
        print(log_string, end='')
        self.log_file.write(log_string)
        time_start = time.time()
        time_elapsed = time_start - self.start_time
        log_string = '[INFO] time:{} Started computing test metrics on the whole dataset\n'.format(
            time_elapsed)
        print(log_string, end='')
        self.log_file.write(log_string)
        nb_file = 4
        result = [0] * nb_file
        threads = []
        # Start computing test metrics
        for i in range(nb_file):
            if i == 0:
                str_ = ''
            else:
                str_ = '_{}'.format(i)
            thread = Thread(target=self.compute_test_for_file, args=(
                os.environ['TEST_DATA_PATH{}'.format(str_)], weights, result, i, ))
            thread.start()
            threads.append(thread)
        [x.join() for x in threads]

        result = functools.reduce(lambda x, y: (
            x[0] + y[0], x[1] + y[1], x[2] + y[2]), result)
        loss = result[0] / result[2]
        acc = result[1] / result[2]
        curr_time = time.time()
        log_string = '[TEST_FINAL] time: {} loss: {} acc: {} duration: {}\n'.format(
            curr_time - self.start_time, loss, acc, curr_time - time_start)
        print(log_string)
        self.log_file.write(log_string)
        self.complete = True
        log_string = '[END]\n'
        print(log_string, end='')
        self.log_file.write(log_string)
        return

    def compute_test_for_file(self, file_path, weight, result, idx):
        ''' Compute test metrics for a specific test file'''
        acc, loss, count = 0, 0, 0
        with open(file_path, 'r') as test_file:
            for line in test_file:
                entries, label = preprocess_line(
                    line, self.targets_all, self.base_idx)
                loss_pt, acc_pt = compute_loss_acc_pt(
                    entries, label, weight, self.du, self.reg)
                loss += loss_pt
                acc += acc_pt
                count += 1
        log_string = '[Info] Done with test for file {}\n'.format(file_path)
        print(log_string, end='')
        with self.log_file_lock:
            self.log_file.write(log_string)
        result[idx] = (loss, acc, count)

    def SendCompletionSignal(self, request, context):
        '''Completion message: Sent by the coordinator when one worker finishes'''
        self.complete = True
        print('[INFO] received end signal')
        return SVM_pb2.Null()

    def GetUpdate(self, request, context):
        # Update weight function for the synchronous case
        self.update_weight(weight_msg_to_dict(request))
        if self.start >= len(self.data):
            self.start = 0
        self.iter_num = request.iteration_number
        return self.start_computation_worker_synch()

    def Start(self, request, context):
        '''Start the training process in the asychronous mode.This function is called by the coordinators after the workers got initialized '''
        if self.is_worker:
            print('[INFO] Worker {} started computing'.format(self.worker_nb))
            # self.current_thread = Thread(
            #    target=self.start_computation_worker_asynch)
            # self.current_thread.start()

            _thread.start_new_thread(self.start_computation_worker_asynch, ())
            print('STARTED')
        return SVM_pb2.Null()

    def SendNodeInfo(self, request, context):
        '''Discovers other pods on the network. Message sent by coordinators before the training process starts '''
        nb_nodes = int(os.environ['NUMBER_REPLICAS'])
        self.worker_nb = int(os.environ['MY_POD_NAME'].split('-')[1])
        if(int(os.environ['ASNYCH'])):
            all_nodes = list(range(nb_nodes - 1))
            del all_nodes[self.worker_nb]
            for i in all_nodes:
                worker_name = 'sgd-{}.sgd'.format(i)
                worker_channel = '{}:50051'.format(
                    socket.gethostbyname(worker_name))
                self.workers_address.append(worker_channel)
                self.worker_stubs.append(SVM_pb2_grpc.SVMStub(
                    grpc.insecure_channel(worker_channel)))
                print('ADDED  worker {}'.format(i))

        coordinator_name = 'sgd-{}.sgd'.format(nb_nodes - 1)
        self.coordinator_address = '{}:50051'.format(
            socket.gethostbyname(coordinator_name))
        self.coordinator_stub = SVM_pb2_grpc.SVMStub(
            grpc.insecure_channel(self.coordinator_address))

        if self.is_worker:
            print('[INFO] WORKER {} received all addresses'.format(self.worker_nb))
        return SVM_pb2.Null()

    def GetWeights(self, weightUpdate, context):
        '''receives weight indices from other workers, accumulates them and maintains a counter for # of updates received '''
        self.get_weight_helper(weightUpdate)

        return SVM_pb2.Null()

    def get_weight_helper(self, weightUpdate):
        '''receives weight indices from other workers, accumulates them and maintains a counter for # of updates received '''
        worker_nb = weightUpdate.worker_nb
        iter_num = weightUpdate.iteration_number
        weightUpdate = weight_msg_to_dict(weightUpdate)
        if not self.is_worker:
            # if current pod is a coordinator, print logs and to the optimization step
            time_elapsed = time.time() - self.start_time
            log_string = '[worker {}] time: {} it: {} Loss: {} acc: {} weight_size: {} \n'.format(
                worker_nb, time_elapsed, iter_num, weightUpdate[-1], weightUpdate[-2], len(self.params))
            self.log_file.write(log_string)
            print(log_string, end='')
            del weightUpdate[-1]
            del weightUpdate[-2]
            if self._asynch:
                # Coordinator update the weights after every iteration on asynchronous mode
                self.update_weight(weightUpdate)
            else:
                # In synchronous mode the gradients are accumulated and the step is done once the gradient from all workers was received
                add_to(self.rcv_grads, weightUpdate, inplace=True)
            cond = False

            # Check if we need to stop training
            if iter_num < self.tot_iter - 1:
                with self.currently_computed_elem_lock:
                    thresh_test = iter_num - self.currently_computed_elem
                    # Check if validation accuracy needs to be computed
                    if thresh_test >= self.test_after:
                        self.currently_computed_elem = iter_num
                        cond = True
                if cond:
                    #_thread.start_new_thread(self.compute_validation,(iter_num,))
                    self.pool_threads.submit(self.compute_validation, iter_num)

            else:
                # If stopping condition reached, notify all the workers and start computing test accuracy if it was not done before
                if self._asynch:
                    futures = []
                    for worker in self.worker_stubs:
                        futures.append(
                            worker.SendCompletionSignal.future(SVM_pb2.Null()))
                        # worker.SendCompletionSignal.SVM_pb2.Null()
                        #_thread.start_new_thread(worker.SendCompletionSignal,(SVM_pb2.Null(),))
                        # self.pool_threads.submit(worker.SendCompletionSignal,SVM_pb2.Null())
                    [x.result() for x in futures]
                log_string = '[INFO] time: {} Received end signal and sent completion signals to all workers\n' .format(
                    time.time() - self.start_time)
                with self.log_file_lock:
                    self.log_file.write(
                        log_string)
                print(log_string, end='')

                #_thread.start_new_thread(self.compute_test, ())
                self.pool_threads.submit(self.compute_test)

            if time.time() - self.time_flush > 15:
                self.log_file.flush()
                self.time_flush = time.time()

            # print('[INFO] Worker {} received vector from worker {}'.format(
            #     self.worker_nb, worker_nb))

        else:
            # If the current pod is a worker accumulate the gradient until next iteration
            add_to(self.rcv_grads, weightUpdate,
                   inplace=True)
            print('[INFO] Worker {} received vector from worker {}'.format(
                self.worker_nb, worker_nb))
        return

    def update_weight(self, grad):
        # Function responsible for the optimization step
        add_to(self.params, scalar_vec_mul(-self.lr, grad),
               inplace=True)
        return

    def start_computation_worker_asynch(self):
        '''Main training function for the asynch mode which is executed independently at every worker'''
        random_indices = list(range(len(self.data)))

        iter_num = 0
        start, end = 0, self.batch_size
        epoch = 0

        future_tmp = None
        # Continue training while we have not reached the stopping condition and  completion signal was not received
        while iter_num < self.tot_iter and (not self.complete):
            # training code
            if start >= len(self.data):
                start, end = 0, self.batch_size
                epoch += 1
            grad, acc, loss = compute_gradient(
                self, random_indices[start:end])
            grad_msg = dict_to_weight_msg(grad)
            grad_msg.iteration_number = iter_num
            grad_msg.worker_nb = self.worker_nb
            for stub in self.worker_stubs:
                #thread = Thread(target=stub.GetWeights, args=(grad_msg,))
                # thread.start()
                stub.GetWeights.future(grad_msg)
                #_thread.start_new_thread(stub.GetWeights, (grad_msg,))
                # self.pool_threads.submit(stub.GetWeights,grad_msg)
            # Build gradient message
            grad_msg.entries.extend([SVM_pb2.Entry(index=-1, value=loss)])
            grad_msg.entries.extend([SVM_pb2.Entry(index=-2, value=acc)])
            # thread = Thread(
            #     target=self.coordinator_stub.GetWeights, args=(grad_msg,))
            # thread.start()
            future_tmp = self.coordinator_stub.GetWeights.future(grad_msg)
            # _thread.start_new_thread(
            #     self.coordinator_stub.GetWeights, (grad_msg,))
            # self.pool_threads.submit(self.coordinator_stub.GetWeights.future,grad_msg)

            # Add currrent gradient to the buffered onces and do the optimization step locally at the current worker

            add_to(grad, self.rcv_grads.copy(), inplace=True)

            self.update_weight(grad)

            self.rcv_grads = {}

            iter_num += 1


            start += self.batch_size
            end += self.batch_size

        self.complete = True
        future_tmp.result()
        print('[INFO] DONE!', iter_num)


    def compute_validation(self, iter_num):
        # Function that computes validation metrics
        print('[INFO] Started computing validation Loss and acc')
        loss, acc = compute_loss_acc(self, self.data, self.target)
        time_elapsed = time.time() - self.start_time
        log_string = '[TEST] time: {} it: {} Loss: {} acc: {} \n'.format(
            time_elapsed, iter_num, loss, acc)
        with self.log_file_lock:
            self.log_file.write(
                log_string)
        print(log_string, end='')
        return

    def start_computation_worker_synch(self):
        # Function that computes the gradient at every worker in sychronous mode
        start, end = self.start, self.start + self.batch_size
        if end > len(self.data):
            end = len(self.data)

        grad, acc, loss = compute_gradient(self, range(start, end))
        grad_msg = dict_to_weight_msg(grad)
        grad_msg.iteration_number = self.iter_num
        grad_msg.worker_nb = self.worker_nb

        grad_msg.entries.extend([SVM_pb2.Entry(index=-1, value=loss)])
        grad_msg.entries.extend([SVM_pb2.Entry(index=-2, value=acc)])

        # self.coordinator_stub.GetWeights(grad_msg)
        self.start += self.batch_size

        return grad_msg
