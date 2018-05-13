import numpy as np
import pickle


with open('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/word2int.pkl', 'rb') as f:
    word2int = pickle.load(f)
with open('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/int2word.pkl', 'rb') as f:
    int2word = pickle.load(f)

train_y =      np.load('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/rating_train.npy')
validation_y = np.load('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/rating_validation.npy')
test_y =       np.load('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/rating_test.npy')

hash_key = tf.convert_to_tensor([key.decode('utf-8') for key in word2int.keys()], tf.string)
hash_value = tf.convert_to_tensor([word2int.get(key) for key in word2int.keys()], tf.int32)

table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(hash_key, hash_value), 1)

def make_feed(mode, table):
    
    train_dataset = tf.data.TextLineDataset(file_path)
    train_dataset = train_dataset.map(lambda string: tf.string_split([string]).values) #Sparse tensor to Dense tensor
    train_dataset = train_dataset.map(lambda words: (words, tf.size(words)))
    train_dataset = train_dataset.map(lambda words, size: (table.lookup(words), size))
    train_dataset = train_dataset.padded_batch(hparams.batch_size, (tf.TensorShape([None]),tf.TensorShape([])))

    train_y =      np.load('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/rating_train.npy')
    train_rating = tf.data.Dataset.from_tensor_slices(train_y)
    train_rating = train_rating.batch(hparams.batch_size)
    train_dataset = tf.data.Dataset.zip((train_dataset, train_rating))
    train_dataset = train_dataset.map(lambda (inputs, size), labels : (inputs, labels, size) )
    data_iter  = train_dataset.make_initializable_iterator()
    inputs, labels, size = data_iter.get_next()
    initializer = data_iter.initializer
    
    return inputs, labels, size, initializer