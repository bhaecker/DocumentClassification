from keras import backend as K
from tensorflow.python.client import device_lib
print(K.tensorflow_backend._get_available_gpus())
print(device_lib.list_local_devices())