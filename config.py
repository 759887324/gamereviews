import os

num_classes = 6

base_dir = './'
data_dir = os.path.join(base_dir, 'data')

train_dir = os.path.join(data_dir, '')
val_dir = os.path.join(data_dir, '')
test_dir = os.path.join(data_dir, '')
vocab_dir = os.path.join(data_dir, 'cail_vocab')
pretrain_dir = os.path.join(data_dir, 'zh.bin')

save_dir = os.path.join(base_dir, 'checkpoints')
save_path = os.path.join(save_dir, '{}/model')
result_path = os.path.join(base_dir, '{}/result')