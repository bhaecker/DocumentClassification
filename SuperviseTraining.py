import numpy as np

from matplotlib import pyplot as plt


history=np.load('history_topBlocks.npy',allow_pickle='TRUE').item()

print(history)

plt.plot(history.history['loss'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
