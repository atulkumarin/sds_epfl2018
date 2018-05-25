import os

lr = 0.1
val_split = 0.15
tot_iter = 5000
batch_size = 128
reg = 0.01
_async = True  # True or False
val_data_size = 2000
test_after = 50
train_data_path = 'ccat_train_data.dat'
test_data_path = 'ccat_test_data.dat'
log_file_path = 'out.dat'
proba_sample = 0.5
conf_dic = {'NUMBER_REPLICAS':'4',}
