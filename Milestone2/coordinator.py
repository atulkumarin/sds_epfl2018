import grpc
from concurrent import futures
import sys
import SVM_pb2
import SVM_pb2_grpc
from server import SVMServicer

##
import configuration as config
##

    
if __name__ == '__main__':
    
    '''Outline :
        load configuration file : ip addresses + port, data path, learning configuration
        Loading Phase:
            send port information to all workers
            send data to all workers
            send learning config signal to all workers
    '''

    port = sys.argv[1]
    coordinator = SVMServicer(is_worker=False)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
    SVM_pb2_grpc.add_SVMServicer_to_server(coordinator, server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    
    #Creating stubs for each worker channel
    #Send channel information to each worker
    coordinator.worker_stubs = []
    
    for channel in config.channels:
        stub = SVM_pb2_grpc.SVMStub(grpc.insecure_channel(channel))
        coordinator.worker_stubs.append(stub)
        
        msg = SVM_pb2.Node_Config()
        msg.coordinator_address=config.coordinator_channel
        
        to_send = config.channels.copy()
        to_send.remove(channel)
        
        msg.workers_address = to_send
       
        _ = stub.SendNodeInfo(msg)
        
    #Send data to all workers
    #writecode
    
    #Send learning configuration   
    for stub in coordinator.worker_stubs:
        msg = SVM_pb2.Learning_Config(lr=config.lr, tot_iter=config.tot_iter, batch_size=config.batch_size, reg=config.reg)
        _ = stub.SendLearningInfo(msg)
        
        
    try:
        while(coordinator.complete is not True):
            pass
        
        fig, ax = plt.subplots(1,2)
        #ax[0].plot(list(range(0, 20*len(test_losses), 20)), losses)
        ax[0].plot(list(range(len(losses))), losses)
        #ax[0].legend(['Validation', 'Train'])
        #ax[1].plot(list(range(0, 20*len(test_accuracies), 20)), test_accuracies)
        ax[1].plot(list(range(len(accuracies))), accuracies)
        #ax[1].legend(['Validation', 'Train'])
        plt.savefig("train.png")
        print('Plot Saved')
        #print('Time = {}'.format(time.time() - start))
        #print(len(test_losses), len(losses))
        
        
    #    out_file_train = open('Out_log_train.txt','w')
    #    out_file_test =open('Out_log_test.txt','w')
    
        '''    
                nb_workers = int(sys.argv[1]) if len(sys.argv)>1  else 3
                lr = 0.1
                reg_factor = 0
                responses = [None] * nb_workers
                for i in range(nb_workers):
                    
                    channels.append(50051 + i)
                try:
                    run()
                except KeyboardInterrupt:
                    print(len(test_losses),len(test_accuracies),len(losses))
                    fig, ax = plt.subplots(1,2)
                    ax[0].plot(list(range(0, 20*len(test_losses), 20)), test_losses)
                    ax[0].plot(list(range(len(losses))), losses)
                    ax[0].legend(['Validation', 'Train'])
                    ax[1].plot(list(range(0, 20*len(test_accuracies), 20)), test_accuracies)
                    ax[1].plot(list(range(len(accuracies))), accuracies)
                    ax[1].legend(['Validation', 'Train'])
                    plt.savefig("train.png")
                    print('Plot Saved')
                    print('Time = {}'.format(time.time() - start))
                    print(len(test_losses), len(losses)) 
        '''
        
    
        
        
    
        
    
    
