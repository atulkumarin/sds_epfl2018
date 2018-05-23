channels = ['localhost:{}'.format(i) for i in range(50050, 50053)]
coordinator_channel = 'localhost:50059'

lr = 0.1
val_split = 0.15
tot_iter = 100
batch_size = 32
reg = 0.1
_async = True  # True or False
val_data_size = 10000
test_after =0.25*(23000/(3*batch_size))
train_data_path = 'data_sds/ccat_train_data.dat'
test_data_path = 'data_sds/ccat_test_data.dat'
log_file_path = 'OUT.txt'
