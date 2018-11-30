configuration = {
    'model_path':'checkpoint',
    'dataset_dir':'data/tfrecord/',

    'num_gpus':1,

    'num_examples_per_epoch_train':4000,
    'num_examples_per_epoch_eval':1000,
    'num_epoches':10000,
    'train_steps':200000,
    'batch_size':32,
    'learning_rate':1e-4,
    'weight_decay':2e-4,
    'momentum':0.9,

}