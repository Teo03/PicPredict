# add or remove classes for download and inference
CLASSES = [
    'eye',
    'bicycle',
    'tree',
    'alarm_clock',
    'book',
    'airplane',
    'cell_phone',
    'smiley_face',
    'apple',
    'car'
]

# configure model params
MODEL_CFG = {
    'epochs': 30,
    'batch_size': 128,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lr_decay_step': [12, 20],
    'gamma': 0.001
}

# configure image processing params
IMAGE_SIZE = 28
TEST_PCT = 0.2
ITEMS_PER_CLASS = 5000

# save locations
NPY_DIR = 'raw'
DATA_DIR = 'data'
MODELS_DIR = 'models'