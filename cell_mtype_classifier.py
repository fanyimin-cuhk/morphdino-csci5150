
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

root_dir = '/mnt/data/aim/liyaxuan/projects/project2/M1_EXC/'

train_labels = np.load(root_dir + 'train_labels.npy')
train_data = np.load(root_dir + 'train_data.npy')

val_labels = np.load(root_dir + 'val_labels.npy')
val_data = np.load(root_dir + 'val_data.npy')

k_nn = KNeighborsClassifier(n_neighbors=5)
k_nn.fit(train_data, train_labels)

predicted_labels = k_nn.predict(train_data)
acc_num = np.sum((train_labels == predicted_labels) + 0)
total_num = len(train_labels)

#print('Training score: ', k_nn.score(train_data, train_labels))
print('Training Acc: ', acc_num * 1.0 / total_num)

#print('Test score: ',  k_nn.score(val_data, val_labels))

predicted_labels = k_nn.predict(val_data)
acc_num = np.sum((val_labels == predicted_labels) + 0)
total_num = len(val_labels)
print('Test Acc:', acc_num * 1.0 / total_num)
