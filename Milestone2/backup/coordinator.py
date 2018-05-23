import grpc
from concurrent import futures
import sys
import SVM_pb2
import SVM_pb2_grpc
from server import SVMServicer
from utils import *
from time import sleep
from threading import Thread

##
import configuration as config

##


class Coordinator():
    def start_(self,address,list_elems):

        '''Outline :
            load configuration file : ip addresses + port, data path, learning configuration
            Loading Phase:
                send port information to all workers
                send data to all workers
                send learning config signal to all workers
        '''

        coordinator = SVMServicer(is_worker=False)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        SVM_pb2_grpc.add_SVMServicer_to_server(coordinator, server)
        coordinator.my_server = server
        server.add_insecure_port(address)
        server.start()
        list_elems.append(server)


        # Creating stubs for each worker channel
        # Send channel information to each worker
        coordinator.worker_stubs = []
        futures_ = []
        for idx,channel in enumerate(config.channels):
            channel_obj = grpc.insecure_channel(channel)
            stub = SVM_pb2_grpc.SVMStub(channel_obj)
            coordinator.worker_stubs.append(stub)

            msg = SVM_pb2.Node_Config()
            msg.coordinator_address = config.coordinator_channel

            to_send = config.channels.copy()
            del to_send[idx]

            msg.workers_address.extend(to_send)
            msg.worker_nb = idx
            grpc.channel_ready_future(channel_obj).result()
            print('[INFO] Worker {} ready'.format(idx))
            futures_.append(stub.SendNodeInfo.future(msg))

        [x.result() for x in futures_]
        #[for x in coordinator.worker_stubs]
        
        return coordinator

    def start_train(self, coordinator): 
        
        if coordinator._async:
        
            threads=[]

            msg = SVM_pb2.Null()

            for worker in coordinator.worker_stubs:
                thread = Thread(target=worker.Start,args=(msg,))
                thread.start()
                threads.append(thread)
                #worker.Start.future(SVM_pb2.Null())

            [x.join() for x in threads]

        else:
            futures_ = []
            for worker in coordinator.worker_stubs:
                grad_msg = dict_to_weight_msg(coordinator.rcv_grads)
                grad_msg.iteration_number = -1
                grad_msg.worker_nb = -1
                futures_.append(worker.GetUpdate.future(grad_msg))

            [x.result() for x in futures_]

            while coordinator.iter_num < coordinator.tot_iter:
                if coordinator.sync_num_rcvgrads == len(coordinator.worker_stubs):
                    coordinator.sync_num_rcvgrads = 0
                    coordinator.iter_num += 1
                     
                    grad_msg = dict_to_weight_msg(coordinator.rcv_grads)
                    coordinator.rcv_grads = {}
                    grad_msg.iteration_number = -1
                    grad_msg.worker_nb = -1      
                    
                    futures_ = []
                    for worker in coordinator.worker_stubs:
                        #grad_msg = dict_to_weight_msg(coordinator.rcv_grads)
                        #grad_msg.iteration_number = -1
                        #grad_msg.worker_nb = -1
                        futures_.append(worker.GetUpdate.future(grad_msg))

                    [x.result() for x in futures_]

        #while(not coordinator.complete):
            #sleep(30)
