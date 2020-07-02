import os
# file containing config parameters fetched from the current context
lr = float(os.environ['lr'])
tot_iter = int(os.environ['tot_iter'])
batch_size = int(os.environ['batch_size'])
reg = float(os.environ['reg'])
val_data_size = int(os.environ['val_data_size'])
test_after = int(os.environ['test_after'])
log_file_path = os.environ['LOG_FILE']
proba_sample = float(os.environ['proba_sample'])
