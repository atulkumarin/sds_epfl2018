import os

lr = 0.1
val_split = 0.15
tot_iter = 5000
batch_size = 128
reg = 0.01
_async = True  # True or False
val_data_size = 2000
test_after = 50
train_data_path = os.environ['TRAIN_FILE_PATH']
test_data_path = os.environ['TEST_FILE_PATH']
log_file_path = os.environ['LOG_FILE']
proba_sample = 0.5
