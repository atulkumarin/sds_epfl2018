channels = ['localhost:{}'.format(i) for i in range(50050, 50053)]
coordinator_channel = 'localhost:50059'

lr = 0.1
val_split = 0.15
tot_iter = 100
batch_size = 128
reg = 0.01
_async = True  # True or False
val_data_size = 2000
test_after = 50
train_data_path = 'data_sds/ccat_train_data.dat'
test_data_path = 'data_sds/ccat_test_data.dat'
log_file_path = 'OUT.txt'
