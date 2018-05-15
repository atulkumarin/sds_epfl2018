from concurrent import futures
import grpc
import sys
import SVM_pb2
import SVM_pb2_grpc
import random
from server import SVMServicer

def serve(port):
    '''starts the server	'''
    worker = SVMServicer(is_worker=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
    SVM_pb2_grpc.add_SVMServicer_to_server(worker, server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    
    
    try:
        iter_num = 0
        
        while iter_num < worker.total_iter: #change for stopping criteria
            #training start
            #select random data points (=batch_size)
            random_indices = random.sample(range(len(worker.data)), worker.batch_size)            
            grad = worker.compute_gradient(random_indices)
            
            grad_msg = worker.dict_to_weight_msg(grad)
            
            for stub in worker.worker_stubs:
                _ = stub.GetWeights(grad_msg)
                
            _ = worker.coordinator_stub.GetWeights(grad_msg)
                               
            #synchornous - check if updates received from all other workers
            #this is done by checking the number of updates received
            
            if(~worker.async):
                while(worker.num_rcv_grads != len(worker.worker_stubs)):
                    pass
            
            total_grad = worker.add_to(grad, worker.rcv_grads)
            
            worker.update_weight(total_grad)
            
            worker.rcv_grads = {}
            worker.num_rcv_grads = 0
            
            for stub in worker.worker_stubs:
                _ = stub.UpdateSignal(SVM_pb2.Null())
            
            if(~worker.async):
                while(worker.num_nodes_update != len(worker.worker_stubs)):
                    pass
            
            worker.num_nodes_update = 0
            
            
            loss, acc = worker.compute_loss_acc(worker.data, worker.target)
            worker.losses.append[loss]
            worker.acc.append[acc]
            iter_num += 1
            
        _ = worker.coordinator_stub.GetWeights(SVM_pb2.Null())
            

    except KeyboardInterrupt:

        server.stop(0)

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("workers", type=int,
                    help="Numbers of workers",default = 3,nargs='?')
    args = parser.parse_args()
    '''

    '''
    nb_workers =  int(sys.argv[1]) if len(sys.argv)>1  else 3
	threads = []
	port = 50051

	for i in range(nb_workers):

		thread = Thread(target = serve, args = (port, ))
		thread.start()
		threads.append(thread)
		port += 1

	for thread in threads:

		thread.join()
    '''
    
    serve(sys.argv[1])
