params = dict()

params['num_classes'] = 20

params['dataset'] = '../../Data/VIRAT/slowfast_clips'

params['epoch_num'] = 40
params['batch_size'] = 32 
params['step'] = 10
params['num_workers'] = 8
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
params['pretrained'] = None
params['gpu'] = [1]
params['log'] = 'log'
params['save_path'] = 'VIRAT'
params['clip_len'] = 64
params['frame_sample_rate'] = 1
params['patience'] = 5
