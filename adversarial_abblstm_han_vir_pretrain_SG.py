import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
from utils.prepare_data import *
from tensorflow.python.client import device_lib
from data_process import *
import collections
from collections import defaultdict
from bs4 import BeautifulSoup
import sys, re
import os
import nltk
from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import math
from data_prepare import gen_vocab
from data_prepare import gen_id_seqs

import argparse
# originate 35,100,256 50000vocab
parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--sen', metavar='S', default=35, type=int,
                   help='Number of Max Sent (Default: 3)')
parser.add_argument('--wrd', metavar='W', default=100, type=int,
                   help='Number of Max Word (Default: 30)')
parser.add_argument('--hid', metavar='H', default=256, type=int,
                   help='Number of Inner Embedding (Default: 100)')
parser.add_argument('--emb', metavar='EMB', default=256, type=int,
                   help='Number of Embedding size (Default: 100)')
parser.add_argument('--drp', metavar='D', default=0.9, type=float,
                   help='Dropout (default: 0.5)')
parser.add_argument('--eps', metavar='E', default=0.5, type=float,
                   help='conv1d (default: 100)')
parser.add_argument('--eps2', metavar='E2', default=0.5, type=float,
                   help='conv1d (default: 100)')
parser.add_argument('--gpu', metavar='G', default=1, type=int,
                   help='gpu device(default:0)')
parser.add_argument('--voc', metavar='V', default=50000, type=int,
                   help='Vocab(default:50000)')

args = parser.parse_args()
GRU_DIM = args.hid
DROPOUT = args.drp
EMBEDDING_DIM = args.emb
EPSILON = args.eps
EPSILON2 = args.eps2
MAX_SENTS = args.sen
MAX_SENT_LENGTH = args.wrd
GPU = args.gpu
if GPU == 0:
    GPU_device = ['/gpu:0']
elif GPU == 1:
    GPU_device = ['/gpu:1']
else:
    GPU_device = ['/gpu:0','/gpu:1']
config = tf.ConfigProto()
#config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
config.gpu_options.allow_growth = True

#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
#config.gpu_options.allow_growth = True

#print(device_lib.list_local_devices())
#print(tf.test.gpu_device_name())
# Hyperparameters
MAX_DOCUMENT_LENGTH = MAX_SENTS #5
MAX_WORD_LENGTH = MAX_SENT_LENGTH #30
MAX_VOCAB = args.voc
EMBEDDING_SIZE = EMBEDDING_DIM #256
HIDDEN_SIZE = GRU_DIM #256
BATCH_SIZE = 8
KEEP_PROB = DROPOUT #0.3
epsilon = EPSILON #5.0  # IMDB ideal norm length
epsilon2 = EPSILON2 #5.0  # IMDB ideal norm length
MAX_LABEL = 5
epochs = 50
TRAINING_SIZE = 450000

def keras_process_sogou(MAX_DOCUMENT_LENGTH,MAX_WORD_LENGTH,MAX_VOCAB):
    data_train = pd.read_csv('../../sogou_news_csv/train.csv', sep=',')
    #print data_train.head(10)
    data_test = pd.read_csv('../../sogou_news_csv/test.csv', sep=',')
    #print data_test.head(10)
    data_full = pd.concat([data_train, data_test], axis=0)
    data_full = data_full.reset_index(drop=True)
    #print data_full
    data_full['text']=data_full['text']+'. '+data_full['text2']
    data_full['label']=data_full['label'].apply(lambda x : x-1)
    print data_full.shape
    #data_full = data_full[:10000]
    reviews = []
    labels = []
    texts = []
    s=[]
    w=[]

    for idx in range(data_full.shape[0]):
        sys.stdout.write("\rProcessing ---- %d"%idx)
        sys.stdout.flush()
        text = BeautifulSoup(data_full.text[idx])
        text = clean_str(text.get_text().encode('ascii','ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        #print(data_full.label[idx])
        labels.append(to_one_hot(data_full.label[idx], MAX_LABEL))

    tokenizer = Tokenizer(num_words=MAX_VOCAB,oov_token='<UNK>')
    tokenizer.fit_on_texts(texts)
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_VOCAB} # <= because tokenizer is 1 indexed
    if len(tokenizer.word_index)>MAX_VOCAB:
        tokenizer.word_index[tokenizer.oov_token] = MAX_VOCAB + 1
    else:
        tokenizer.word_index[tokenizer.oov_token] = len(tokenizer.word_index) 

    data = np.zeros((len(texts), MAX_DOCUMENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')
    with open("data/SG_train.ids", "w") as train_file, open("data/SG_test.ids", "w") as test_file:
        for i, sentences in enumerate(reviews):
            sequence = tokenizer.texts_to_sequences(sentences)
            padded_sequence = pad_sequences(sequence, maxlen=MAX_SENT_LENGTH, dtype='int32',padding='post', truncating='post', value=0.)
            #current_file.write(" ".join([str(id) for id in padded_sequence]) + "\n")
            #print(sequence)                   
            #print(padded_sequence)
            s.append(len(sequence))              
            for k, seq in enumerate(sequence):
                #print(seq)
                w.append(len(seq))                    
            for j, sent in enumerate(sentences):
                if j< MAX_DOCUMENT_LENGTH:
                    data[i,j] = padded_sequence[j]
                if i< TRAINING_SIZE:
                    train_file.write(" ".join([str(id) for id in padded_sequence[j]]) + "\n")
                else:
                    test_file.write(" ".join([str(id) for id in padded_sequence[j]]) + "\n")
                    #wordTokens = text_to_word_sequence(sent)
                    #k=0
                    #for _, word in enumerate(wordTokens):
                    #    if k<MAX_WORD_LENGTH and tokenizer.word_index[word]<MAX_VOCAB:
                            #data[i,j,k] = tokenizer.word_index[word]
                    #        k=k+1
                    #    else:
                    #        print('word',word)
                    #print('sentence',wordTokens)
                    #print(data[i,j])
                    #sequence = tokenizer.texts_to_sequences(sentences)
                    #padded_sequence = pad_sequences(sequence, maxlen=MAX_SENT_LENGTH, dtype='int32',padding='post', truncating='post', value=0.)    
                    #print(sequence)                   
                    #print(padded_sequence)    

    s_des = np.percentile(s, [25, 50, 75, 90, 98])
    print ('25,, of sentences found',s_des[0])
    print (',50, of sentences found',s_des[1])
    print (',,75 of sentences found',s_des[2])
    print (',,90 of sentences found',s_des[3])
    print (',,98 of sentences found',s_des[4])
    w_des = np.percentile(w, [25, 50, 75, 90, 98])
    print ('25,, of words found',w_des[0])
    print (',50, of words found',w_des[1])
    print (',,75 of words found',w_des[2])
    print (',,90 of words found',w_des[3])
    print (',,98 of words found',w_des[4])               
                    
    word_index = tokenizer.word_index
    word_docs = tokenizer.word_docs
    word_counts = tokenizer.word_counts
    #print('word_index',len(word_index))
    #print('word_counts',len(word_counts))
    #print(word_counts)
    #print('word_counts a',word_counts['a'])
    #print('word_counts 10001',word_counts['999'])
    #print('word_counts 10001',word_counts['999'])
    #print('word_counts 10001',word_counts['10000'])
    #print('word_counts 10001',word_counts['10001'])
    x_train = np.array(data[:TRAINING_SIZE])
    x_test = np.array(data[TRAINING_SIZE:])
    y_train = np.array(labels[:TRAINING_SIZE])
    y_test = np.array(labels[TRAINING_SIZE:])
    #x_train = np.array(data)
    #x_test = np.array(data)
    #y_train = np.array(labels)
    #y_test = np.array(labels)
    return x_train, x_test, y_train, y_test, word_counts, word_index, len(word_index)+1

def keras_process_AG(MAX_DOCUMENT_LENGTH,MAX_WORD_LENGTH,MAX_VOCAB):
    data_full = pd.read_csv('../../AG_news/train.csv', sep=',')
    #data_full = pd.read_csv('../../AG_news/train_m.csv', sep=',')
    print data_full.shape

    reviews = []
    labels = []
    texts = []
    s=[]
    w=[]

    for idx in range(data_full.shape[0]):
        text = BeautifulSoup(data_full.text[idx])
        text = clean_str(text.get_text().encode('ascii','ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        #print(data_full.label[idx])
        labels.append(to_one_hot(data_full.label[idx], 4))

    tokenizer = Tokenizer(num_words=MAX_VOCAB,oov_token='<UNK>')
    tokenizer.fit_on_texts(texts)
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_VOCAB} # <= because tokenizer is 1 indexed
    if len(tokenizer.word_index)>MAX_VOCAB:
        tokenizer.word_index[tokenizer.oov_token] = MAX_VOCAB + 1
    else:
        tokenizer.word_index[tokenizer.oov_token] = len(tokenizer.word_index) 

    data = np.zeros((len(texts), MAX_DOCUMENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')
    with open("data/SG_train.ids", "w") as train_file, open("data/SG_test.ids", "w") as test_file:
        for i, sentences in enumerate(reviews):
            sequence = tokenizer.texts_to_sequences(sentences)
            padded_sequence = pad_sequences(sequence, maxlen=MAX_SENT_LENGTH, dtype='int32',padding='post', truncating='post', value=0.)
            #current_file.write(" ".join([str(id) for id in padded_sequence]) + "\n")
            #print(sequence)                   
            #print(padded_sequence)
            s.append(len(sequence))              
            for k, seq in enumerate(sequence):
                #print(seq)
                w.append(len(seq))                    
            for j, sent in enumerate(sentences):
                if j< MAX_DOCUMENT_LENGTH:
                    data[i,j] = padded_sequence[j]
                if i< TRAINING_SIZE:
                    train_file.write(" ".join([str(id) for id in padded_sequence[j]]) + "\n")
                else:
                    test_file.write(" ".join([str(id) for id in padded_sequence[j]]) + "\n")
                    #wordTokens = text_to_word_sequence(sent)
                    #k=0
                    #for _, word in enumerate(wordTokens):
                    #    if k<MAX_WORD_LENGTH and tokenizer.word_index[word]<MAX_VOCAB:
                            #data[i,j,k] = tokenizer.word_index[word]
                    #        k=k+1
                    #    else:
                    #        print('word',word)
                    #print('sentence',wordTokens)
                    #print(data[i,j])
                    #sequence = tokenizer.texts_to_sequences(sentences)
                    #padded_sequence = pad_sequences(sequence, maxlen=MAX_SENT_LENGTH, dtype='int32',padding='post', truncating='post', value=0.)    
                    #print(sequence)                   
                    #print(padded_sequence)    

    s_des = np.percentile(s, [25, 50, 75, 90, 98])
    print ('25,, of sentences found',s_des[0])
    print (',50, of sentences found',s_des[1])
    print (',,75 of sentences found',s_des[2])
    print (',,90 of sentences found',s_des[3])
    print (',,98 of sentences found',s_des[4])
    w_des = np.percentile(w, [25, 50, 75, 90, 98])
    print ('25,, of words found',w_des[0])
    print (',50, of words found',w_des[1])
    print (',,75 of words found',w_des[2])
    print (',,90 of words found',w_des[3])
    print (',,98 of words found',w_des[4])               
                    
    word_index = tokenizer.word_index
    word_docs = tokenizer.word_docs
    word_counts = tokenizer.word_counts
    #print('word_index',len(word_index))
    #print('word_counts',len(word_counts))
    #print(word_counts)
    #print('word_counts a',word_counts['a'])
    #print('word_counts 10001',word_counts['999'])
    #print('word_counts 10001',word_counts['999'])
    #print('word_counts 10001',word_counts['10000'])
    #print('word_counts 10001',word_counts['10001'])
    x_train = np.array(data[:TRAINING_SIZE])
    x_test = np.array(data[TRAINING_SIZE:])
    y_train = np.array(labels[:TRAINING_SIZE])
    y_test = np.array(labels[TRAINING_SIZE:])
    #x_train = np.array(data)
    #x_test = np.array(data)
    #y_train = np.array(labels)
    #y_test = np.array(labels)
    return x_train, x_test, y_train, y_test, word_counts, word_index, len(word_index)+1

#dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia', test_with_fake_data=False)
#print(dbpedia)

# load data
#x_train, y_train = load_data("../dbpedia_data/dbpedia_csv/train.csv", sample_ratio=0.1)
#x_train, y_train = load_data_AG("../../AG_news/seperated/train.csv", sample_ratio=1)
#print(x_train)
#print(y_train)
#x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/test.csv", sample_ratio=0.1)
#x_test, y_test = load_data_AG("../../AG_news/seperated/test.csv", sample_ratio=1)

#x_train, x_test, dictionary, vocab_freq = process_AG()
#word2idx, index_to_embedding = load_embedding_from_disks('../../nextlabs/Glove/glove.6B.200d.txt', dictionary, with_indexes=True)
#vocab_size = len(word2idx)

# data preprocessing
#x_train, x_test, y_train, y_test, vocab_freq, word2idx, vocab_size = \
#    data_preprocessing_with_dict(x_train, x_test, MAX_DOCUMENT_LENGTH)

x_train, x_test, y_train, y_test, vocab_freq, word2idx, vocab_size = \
    keras_process_sogou(MAX_DOCUMENT_LENGTH,MAX_WORD_LENGTH,MAX_VOCAB)
print('vocab size',vocab_size)
dictionary = word2idx
#print('x_train',x_train)
#print('x_test',x_test)
#print('y_train',y_train)
#print('y_test',y_test)
#print("Vocab size: ", vocab_size)
#print(vocab_freq)

# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 1)
#print("Validation size: ", dev_size)

def load_embedding_from_disks(glove_filename, dictionary, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct 
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = np.array([np.random.rand(200,)*0.2 for i in range(len(dictionary)+1)])
        #print(index_to_embedding_array)
    else:
        word_to_embedding_dict = dict()

    
    with open(glove_filename, 'r') as glove_file:
        j = 0
        for (i, line) in enumerate(glove_file):
            
            split = line.split(' ')
            
            word = split[0]

            if word in dictionary:

                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )
                
                index = dictionary[word]
                #if (index<10):
                    #print(word)
                    #print(index)
                #print(representation)
                index_to_embedding_array[index] = representation
                

        index_to_embedding_array = np.array(index_to_embedding_array)
        return word_to_index_dict, index_to_embedding_array


#x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/testgg.csv", sample_ratio=1)
def get_freq(vocab_freq, word2idx):
    """get a frequency dict format as {word_idx: word_freq}"""
    words = word2idx.keys()
    freq = [0] * (vocab_size)
    #print('vocab_size',vocab_size)
    #print('vocab_size',len(word2idx))
    for word in words:
        word_idx = word2idx.get(word)
        if word == '<UNK>':
            word_freq = 3
            #print(word_idx)
            freq[word_idx] = word_freq
        else:
            word_freq = vocab_freq[word]
            #print(word_idx)
            freq[word_idx] = word_freq
    return freq

'''
def get_freq(vocab_freq, word2idx,dictionary):
    """get a frequency dict format as {word_idx: word_freq}"""
    words = dictionary.keys()
    vocab_size = len(word2idx)
    print('word2idx',len(word2idx))
    freq = [0] * vocab_size
    for word in words:
        word_idx = word2idx[word]
        #print(word)
        #print(word2idx[word])
        word_freq = vocab_freq[word]
        #print(word_idx)
        freq[word_idx] = word_freq
    return freq
'''

def _scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def add_perturbation(embedded, loss):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, epsilon)
    return embedded + perturb


def normalize(emb, weights):
    # weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
    #print("Weights: ", weights)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev

#v_freq = get_freq(vocab_freq, word2idx)
#print(v_freq)
graph = tf.Graph()
with graph.as_default():
    global dictionary
    lm_batch_size = 64
    num_epochs = 80
    check_point_step = 100
    #num_train_samples = tf.placeholder(tf.int32, name="num_train_samples")
    #num_valid_samples = tf.placeholder(tf.int32, name="num_valid_samples")
    #num_train_samples = num_train_samples
    #num_valid_samples = num_valid_samples
    num_layers = 2
    initial_learning_rate=1.0
    final_learning_rate=0.01
    num_hidden_units = 256
    max_gradient_norm = 5.0
    global_step = tf.Variable(0, trainable=False)
    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH, MAX_WORD_LENGTH])
    #print('batch_x',batch_x)
    #batch_x_print = tf.Print(batch_x,[batch_x],'batch_x : ')
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    #print('batch_y',batch_y)
    keep_prob = tf.placeholder(tf.float32)
    vocab_freqs = tf.constant(get_freq(vocab_freq, word2idx), dtype=tf.float32, shape=(vocab_size, 1))

    weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
    input_embedding_mat = tf.get_variable("input_embedding_mat",
                                                   [vocab_size, num_hidden_units],
                                                   dtype=tf.float32)
    with tf.device('/gpu:0'):

        embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
        #print('embeddings_var',embeddings_var)
    #print(tf.shape(embeddings_var, name='emb_var'))
        W_w_attention_word = tf.Variable(tf.random_normal([HIDDEN_SIZE*2,HIDDEN_SIZE*2], stddev=0.1))
        W_b_attention_word = tf.Variable(tf.random_normal([HIDDEN_SIZE*2], stddev=0.1))
        context_vecotor_word = tf.Variable(tf.random_normal([HIDDEN_SIZE*2], stddev=0.1))
        W_w_attention_sent = tf.Variable(tf.random_normal([HIDDEN_SIZE*2,HIDDEN_SIZE*2], stddev=0.1))
        W_b_attention_sent = tf.Variable(tf.random_normal([HIDDEN_SIZE*2], stddev=0.1))
        context_vecotor_sent = tf.Variable(tf.random_normal([HIDDEN_SIZE*2], stddev=0.1))
        W = tf.Variable(tf.random_normal([HIDDEN_SIZE*2], stddev=0.1))
        W_sent = tf.Variable(tf.random_normal([HIDDEN_SIZE*2], stddev=0.1))
        B = tf.Variable(tf.constant(0., shape=[HIDDEN_SIZE*2]))
        B_sent = tf.Variable(tf.constant(0., shape=[HIDDEN_SIZE*2]))
        W_wfc = tf.Variable(tf.truncated_normal([HIDDEN_SIZE*2, HIDDEN_SIZE*2], stddev=0.1))
        b_wfc = tf.Variable(tf.constant(0., shape=[HIDDEN_SIZE*2]))
        W_fc = tf.Variable(tf.truncated_normal([HIDDEN_SIZE*2, MAX_LABEL], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))

        word2idx, index_to_embedding = load_embedding_from_disks('../../nextlabs/Glove/glove.6B.200d.txt', dictionary, with_indexes=True)
        with tf.device('/cpu:0'):
            EMB = tf.constant(index_to_embedding, name="W_cons")
            EMB = tf.Variable(tf.constant(0.0, shape=index_to_embedding.shape),
                trainable=True, name="W_var")
            tf_embedding_placeholder = tf.placeholder(tf.float32, shape=index_to_embedding.shape)
        sess = tf.Session(config=config)   # sess = tf.Session()

        tf_embedding_init = EMB.assign(tf_embedding_placeholder)
        _ = sess.run(
            tf_embedding_init, 
            feed_dict={
                tf_embedding_placeholder: index_to_embedding
            }
        )
    for i, d in enumerate(GPU_device):
        with tf.device(d):
            embedding_norm = normalize(input_embedding_mat, weights)
            batch_embedded = tf.nn.embedding_lookup(embedding_norm, batch_x)
    print('batch_embedded',batch_embedded)
    #print(tf.shape(batch_embedded, name='emb'))
    def cal_gru_word(batch_embedded, keep_prob, reuse=True, scope="loss_word"):
        for i, d in enumerate(GPU_device):
            with tf.device(d):
                batch_embedded_reshaped = tf.reshape(batch_embedded,[-1,MAX_WORD_LENGTH,EMBEDDING_SIZE])
                with tf.variable_scope('word_gru', reuse=reuse) as scope:
                    wf_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse = tf.AUTO_REUSE) #, reuse = tf.AUTO_REUSE
                    wf_cell = tf.contrib.rnn.DropoutWrapper(wf_cell, input_keep_prob=keep_prob)
            #init_state = wf_cell.zero_state(#batch_size=BATCH_SIZE,
            #                             dtype=tf.float32)  # [batch_size, hidden_size]
                    wf, state = tf.nn.dynamic_rnn(wf_cell, batch_embedded_reshaped, dtype=tf.float32, time_major=False)
            
                    wb_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse = tf.AUTO_REUSE)
            #init_state = wb_cell.zero_state(#batch_size=BATCH_SIZE,
            #                             dtype=tf.float32)  # [batch_size, hidden_size]
                    wb, state = tf.nn.dynamic_rnn(wb_cell, batch_embedded_reshaped, dtype=tf.float32, time_major=False)
                    wb = tf.reverse(wb, [2])

            #rnn_outputs, _ = bi_rnn(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE,reuse=tf.AUTO_REUSE), tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE),inputs=batch_embedded_reshaped, dtype=tf.float32)
            #print(tf.shape(rnn_outputs))
            #print('rnn_outputs',rnn_outputs)
            #hidden_state_forward = gru_forward_word_level(batch_embedded)
            #hidden_state_backward = gru_backward_word_level(batch_embedded)
            #hidden_state = tf.concat([hidden_state_forward, hidden_state_backward], axis=2)
            # Attention
                    H = tf.concat([wf, wb],-1)  # fw + bw
            #H = tf.concat([rnn_outputs[0], rnn_outputs[1]],-1)  # fw + bw
            #print(tf.shape(H, name='H'))
            #H = tf.reshape(H,[-1,256])
            #print('H',H)
                    '''
            M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
            #M = tf.tanh(tf.matmul(H,W) + B)
            #print(tf.shape(M, name='M'))
            #print('M',M)
            #M = tf.reshape(M,[-1,25,256])
            # alpha (bs * sl, 1)
            alpha1 = tf.reshape(M, [-1, HIDDEN_SIZE*2])
            alpha2 = tf.reshape(W, [-1, 1])
            alpha3 = tf.matmul(alpha1,alpha2)
            alpha = tf.nn.softmax(alpha3)
            #print(tf.shape(alpha, name='alpha'))
            #print('alpha1',alpha1)
            #print('alpha2',alpha2)
            #print('alpha3',alpha3)
            #print('alpha',alpha)
            r1 = tf.transpose(H, [0, 2, 1])
            r2 = tf.reshape(alpha, [-1, MAX_WORD_LENGTH,1])
            r = tf.matmul(r1, r2)  # supposed to be (batch_size * HIDDEN_SIZE, 1)
            #print(tf.shape(r, name='r'))
            #print('r1',r1)
            #print('r2',r2)
            #print('r',r)
            r = tf.squeeze(r)
            #print('r',r)
            h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE
            #print('r',h_star)
            # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
            drop = tf.nn.dropout(h_star, keep_prob)
            #print('drop',drop)
            # Fully connected layer dense layer
            y_hat = tf.nn.xw_plus_b(drop, W_wfc, b_wfc)
                    '''
            return H

    
    def cal_loss_logit_word(h_representation, keep_prob, reuse=True, scope="loss_word_att"):
        #with tf.name_scope(scope) as scope:
        for i, d in enumerate(GPU_device):
            with tf.device(d):
                with tf.variable_scope('word_att', reuse=reuse) as scope:

                    hidden_state_2 = tf.reshape(h_representation, shape=[-1,
                                                          HIDDEN_SIZE * 2])
            #print('hidden_state_2',hidden_state_2)
                    hidden_representation_1 = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     W_w_attention_word) + W_b_attention_word)
            #print('hidden_representation_1',hidden_representation_1)

                    hidden_representation_2 = tf.reshape(h_representation, shape=[-1, MAX_WORD_LENGTH,
                                                                         HIDDEN_SIZE * 2])
            #print('hidden_representation_2',hidden_representation_2)
                    hidden_state_context_similiarity = tf.multiply(hidden_representation_2,
                                                       context_vecotor_word)
            #print('hidden_state_context_similiarity',hidden_state_context_similiarity)
                    attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # [batch_size*num_sentences,sentence_length]
            #print('attention_logits',attention_logits)
                    attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # [batch_size*num_sentences,1]
            #print('attention_logits_max',attention_logits_max)
                    p_attention = tf.nn.softmax(
                    attention_logits - attention_logits_max)  # [batch_size*num_sentences,sentence_length]
            #print('attention_logits',attention_logits)
                    p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # [batch_size*num_sentences,sentence_length,1]
            #print('p_attention_expanded',p_attention_expanded)
                    sentence_representation_1 = tf.multiply(p_attention_expanded,
                                              h_representation)  # [batch_size*num_sentences,sentence_length, hidden_size*2]
            #print('sentence_representation_1',sentence_representation_1)
                    sentence_representation_2 = tf.reduce_sum(sentence_representation_1,
                                                axis=1)
            #print('sentence_representation_2',sentence_representation_2)
            #drop = tf.nn.dropout(sentence_representation_2, keep_prob)
            #print('y_hat',y_hat)
                    batch_embedded_reshaped_sent = tf.reshape(sentence_representation_2,[-1,MAX_DOCUMENT_LENGTH,HIDDEN_SIZE*2])
            #print('batch_embedded_reshaped_sent',batch_embedded_reshaped_sent)
            return batch_embedded_reshaped_sent

    def cal_loss_logit_sent(batch_embedded_reshaped_sent, keep_prob, reuse=True, scope="loss_sent"):
        for i, d in enumerate(GPU_device):
            with tf.device(d):
                with tf.variable_scope('sent_gru', reuse=reuse) as scope:
                    sf_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse = tf.AUTO_REUSE)
                    sf_cell = tf.contrib.rnn.DropoutWrapper(sf_cell, input_keep_prob=keep_prob)
            #init_state = wf_cell.zero_state(#batch_size=BATCH_SIZE,
            #                             dtype=tf.float32)  # [batch_size, hidden_size]
                    sf, state = tf.nn.dynamic_rnn(sf_cell, batch_embedded_reshaped_sent, dtype=tf.float32, time_major=False)
            
                    sb_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse = tf.AUTO_REUSE)
            #init_state = wb_cell.zero_state(#batch_size=BATCH_SIZE,
            #                             dtype=tf.float32)  # [batch_size, hidden_size]
                    sb, state = tf.nn.dynamic_rnn(sb_cell, batch_embedded_reshaped_sent, dtype=tf.float32, time_major=False)
                    sb = tf.reverse(sb, [2])
            #rnn_outputs_sent, _ = bi_rnn(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE),inputs=batch_embedded_reshaped_sent, dtype=tf.float32)
            #print(tf.shape(rnn_outputs))
            #print('rnn_outputs_sent',rnn_outputs_sent)
            #hidden_state_forward = gru_forward_word_level(batch_embedded)
            #hidden_state_backward = gru_backward_word_level(batch_embedded)
            #hidden_state = tf.concat([hidden_state_forward, hidden_state_backward], axis=2)
            # Attention
                    H_sent = tf.concat([sf, sb],-1)  # fw + bw
            #H_sent = tf.concat([rnn_outputs_sent[0], rnn_outputs_sent[1]],-1)  # fw + bw
            #print(tf.shape(H, name='H'))
            #H = tf.reshape(H,[-1,256])
            #print('H_sent',H_sent)
                    '''
            M_sent = tf.tanh(H_sent)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
            #M = tf.tanh(tf.matmul(H,W) + B)
            #print(tf.shape(M, name='M'))
            #print('M_sent',M_sent)
            #M = tf.reshape(M,[-1,25,256])
            # alpha (bs * sl, 1)
            alpha1_sent = tf.reshape(M_sent, [-1, HIDDEN_SIZE*2])
            alpha2_sent = tf.reshape(W_sent, [-1, 1])
            alpha3_sent = tf.matmul(alpha1_sent,alpha2_sent)
            alpha_sent = tf.nn.softmax(alpha3_sent)
            #print(tf.shape(alpha, name='alpha'))
            #print('alpha1_sent',alpha1_sent)
            #print('alpha2_sent',alpha2_sent)
            #print('alpha3_sent',alpha3_sent)
            #print('alpha_sent',alpha_sent)
            r1_sent = tf.transpose(H_sent, [0, 2, 1])
            r2_sent = tf.reshape(alpha_sent, [-1, MAX_DOCUMENT_LENGTH,1])
            r_sent = tf.matmul(r1_sent, r2_sent)  # supposed to be (batch_size * HIDDEN_SIZE, 1)
            #print(tf.shape(r, name='r'))
            #print('r1_sent',r1_sent)
            #print('r2_sent',r2_sent)
            #print('r_sent',r_sent)
            r_sent = tf.squeeze(r_sent)
            #print('r_sent',r_sent)
            h_star_sent = tf.tanh(r_sent)  # (batch , HIDDEN_SIZE
            #print('h_star_sent',h_star_sent)
            # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
                    '''
                    hidden_state_2_sent = tf.reshape(H_sent, shape=[-1,
                                                          HIDDEN_SIZE * 2])
                    print('hidden_state_2',hidden_state_2_sent)
                    hidden_representation_1_sent = tf.nn.tanh(tf.matmul(hidden_state_2_sent,
                                                     W_w_attention_sent) + W_b_attention_sent)
            #print('hidden_representation_1',hidden_representation_1_sent)
                    hidden_representation_2_sent = tf.reshape(hidden_representation_1_sent, shape=[-1, MAX_DOCUMENT_LENGTH,
                                                                         HIDDEN_SIZE * 2])
                    print('hidden_representation_2',hidden_representation_2_sent)
                    hidden_state_context_similiarity_sent = tf.multiply(hidden_representation_2_sent,
                                                       context_vecotor_sent)
            #print('hidden_state_context_similiarity',hidden_state_context_similiarity_sent)
                    attention_logits_sent = tf.reduce_sum(hidden_state_context_similiarity_sent,
                                         axis=2)  # [batch_size*num_sentences,sentence_length]
            #print('attention_logits',attention_logits)
                    attention_logits_max_sent = tf.reduce_max(attention_logits_sent, axis=1,
                                             keep_dims=True)  # [batch_size*num_sentences,1]
            #print('attention_logits_max',attention_logits_max_sent)
                    p_attention_sent = tf.nn.softmax(
                    attention_logits_sent - attention_logits_max_sent)  # [batch_size*num_sentences,sentence_length]
            #print('attention_logits',attention_logits)
                    p_attention_expanded_sent = tf.expand_dims(p_attention_sent, axis=2)  # [batch_size*num_sentences,sentence_length,1]
            #print('p_attention_expanded',p_attention_expanded)
                    sentence_representation_1_sent = tf.multiply(p_attention_expanded_sent,
                                              H_sent)  # [batch_size*num_sentences,sentence_length, hidden_size*2]
            #print('sentence_representation_1',sentence_representation_1_sent)
                    sentence_representation_2_sent = tf.reduce_sum(sentence_representation_1_sent,
                                                axis=1)
            #print('sentence_representation_2',sentence_representation_2_sent)
            #drop_sent = tf.nn.dropout(sentence_representation_2_sent, keep_prob)
            #print('drop_sent',drop_sent)
            # Fully connected layer dense layer
                    y_hat_sent = tf.nn.xw_plus_b(sentence_representation_2_sent, W_fc, b_fc)
            #print(tf.shape(y_hat, name='att'))
            #print('y_hat_sent',y_hat_sent)
	#v2
        return y_hat_sent, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat_sent, labels=batch_y))

    def VAT(input_tensor, network_to_approximate=None, xi=1e-6, epsilon=0.5, weight=0.1, num_approximation=1, clip_value_min=1e-30, keep_prob=1.0, dtype=tf.float32):
        '''
    https://arxiv.org/abs/1704.03976
    ===input===
    input_tensor           : input tensor of network
    network                : function that receives input_tensor and returns the logits (i.e., the output without softmax.)
    network_to_approximate : function only to approximate the virtual adversarial perturbation
                             this may be useful when you want network to behave differently from the usual training part at some points such like dropout.
                             if this is None (default), this is same as "network."
    xi                     : scale of perturbation that is used to approximate the virtual adversarial perturbation. (default: 1e-6)
    epsilon                : scale of virtual adversarial perturbation. results can be sensitive at this number. (default: 2.0)
    weight                 : weight of loss. (default: 1.0)
    num_approximation      : number of iteration to approximate the virtual adversarial perturbation. (default: 1)
    clip_value_min         : this is for clipping some values that is divisor or given to log. (default: 1e-30)
    dtype                  : dtype of tensors in this function. (default: tf.float32)
    ===output===
    vat_cross_entropy      : virtual adversarial loss
    vat_perturbation       : virtual adversarial perturbation
    

    if network_to_approximate is None:
        network_to_approximate = network
        isSameNetwork = True
    else:
        isSameNetwork = False

        '''
        isSameNetwork = True
        clipped = lambda x: tf.maximum(x, clip_value_min)

        axis_without_batch_size = tuple(range(1,len(input_tensor.get_shape())))
        if len(axis_without_batch_size) == 1: axis_without_batch_size = axis_without_batch_size[0]
        normalized = lambda x: x / clipped(tf.norm(x, axis=axis_without_batch_size, keep_dims=True))

        plain_softmax = tf.nn.softmax(cal_loss_logit_sent(input_tensor, keep_prob, reuse=True)[0])
        perturbation = xi * normalized(tf.random_normal(shape=tf.shape(input_tensor), stddev=0.1 , dtype=dtype))
        for i in range(num_approximation):
            softmax_accommodating_perturbation = tf.nn.softmax(cal_loss_logit_sent(input_tensor + perturbation, keep_prob, reuse=True)[0])
            cross_entropy_accommodating_perturbation = -tf.reduce_sum(plain_softmax * tf.log(clipped(softmax_accommodating_perturbation)), reduction_indices=1) * weight
            adversarial_direction = tf.gradients(cross_entropy_accommodating_perturbation, [perturbation])[0]
            vat_perturbation = normalized(adversarial_direction)
            perturbation = xi * vat_perturbation

        current_softmax = tf.nn.softmax(cal_loss_logit_sent(input_tensor, keep_prob, reuse=True)[0]) if not isSameNetwork else plain_softmax
        current_softmax = tf.stop_gradient(current_softmax)
        vat_perturbation = tf.stop_gradient(epsilon * vat_perturbation)
        vat_softmax = tf.nn.softmax(cal_loss_logit_sent(input_tensor + vat_perturbation, keep_prob, reuse=True)[0])
        vat_cross_entropy = tf.reduce_sum(-tf.reduce_sum(current_softmax * tf.log(clipped(vat_softmax)), reduction_indices=1) * weight) / (tf.reduce_sum(weight)+1e-30)
        return vat_cross_entropy/100, vat_perturbation

    def logsoftmax(x):
        for i, d in enumerate(GPU_device):
            with tf.device(d):
        #print('x :',x)
                y = tf.reduce_max(x, 0, keep_dims=True)
        #print('y :',y)
                xdev = x - y
        #print('xdev :',xdev)
                lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
        #print('lsm :',lsm)
        return lsm

    def kl_divergence_with_logit(q_logit, p_logit):
        for i, d in enumerate(GPU_device):
            with tf.device(d):
        #print('q_logit :',q_logit)
        #print('p_logit :',p_logit)
                q = tf.nn.softmax(q_logit)
                reduce_sum = tf.reduce_sum(q * logsoftmax(q_logit), 1)
        #print('reduce_sum :',reduce_sum)
                qlogq = tf.reduce_mean(reduce_sum)
        #print('qlogq :',qlogq)
                qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
        #print('qlogp :',qlogp)
        return qlogq - qlogp
    
    def get_normalized_vector(d):
        for i, g in enumerate(GPU_device):
            with tf.device(g):
                d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
                d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
        return d


    def generate_virtual_adversarial_perturbation(x, logit, keep_prob, sent, num_power_iterations=1, xi=1e-6, epsilon=epsilon2):
        d = tf.random_normal(shape=tf.shape(x))

        for _ in range(num_power_iterations):
            d = xi * get_normalized_vector(d)
            logit_p = logit
            if sent == True:
                logit_m = cal_loss_logit_sent(x+d , keep_prob, reuse=True)[0]
            else:
                h_s = cal_gru_word(x+d, keep_prob, reuse=True)
                b_s = cal_loss_logit_word(h_s, keep_prob, reuse=True)
                logit_m = cal_loss_logit_sent(b_s , keep_prob, reuse=True)[0]
            dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
    
        return epsilon * get_normalized_vector(d)


    def virtual_adversarial_loss(x, logit, keep_prob, sent, name='vatloss'):
        r_vadv = generate_virtual_adversarial_perturbation(x, logit, keep_prob, sent)
        logit = tf.stop_gradient(logit)
        logit_p = logit
        if sent == True:
            logit_m = cal_loss_logit_sent(x+ r_vadv, keep_prob, reuse=True)[0]
        else:
            h_s = cal_gru_word(x+ r_vadv, keep_prob, reuse=True)
            b_s = cal_loss_logit_word(h_s, keep_prob, reuse=True)
            logit_m = cal_loss_logit_sent(b_s, keep_prob, reuse=True)[0]
        loss_va = kl_divergence_with_logit(logit_p, logit_m)
        return tf.identity(loss_va, name=name)

    #def model_pretrained():
    #lm_vocab_size = tf.placeholder(tf.int32, name="lm_vocab_size")
    #with open("data/sg_train.ids") as fp:
    #    num_train_samples = len(fp.readlines())
    #with open("data/sg_valid.ids") as fp:
    #    num_valid_samples = len(fp.readlines())

    #with open("data/sg_vocab") as vocab:
    #    lm_vocab_size = len(vocab.readlines())

    # We set a dynamic learining rate, it decays every time the model has gone through 150 batches.
    # A minimum learning rate has also been set.
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           150, 0.95, staircase=True)
    learning_rate = tf.cond(tf.less(learning_rate, final_learning_rate), lambda: tf.constant(final_learning_rate),
                                     lambda: learning_rate)
    output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                    [vocab_size, num_hidden_units*2],
                                                    dtype=tf.float32)

    output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                     [vocab_size],
                                                     dtype=tf.float32)

    #dropout_rate = 0.5 #tf.placeholder(tf.float32, name="dropout_rate")
    dropout_rate = tf.placeholder(tf.float32)

    file_name_train = tf.placeholder(tf.string)
    file_name_validation = tf.placeholder(tf.string)
    file_name_test = tf.placeholder(tf.string)

    def parse(line):
        line_split = tf.string_split([line])
        input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
        #input_seq = tf.Print(input_seq,[input_seq],'input_seq : ')
        output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
        #output_seq = tf.Print(output_seq,[output_seq],'output_seq : ')
        return input_seq, output_seq

    def output_embedding(current_output):
        oputs = tf.add(
            tf.matmul(current_output, tf.transpose(output_embedding_mat)), output_embedding_bias)
        print('oputs:',tf.transpose(output_embedding_mat))
        return oputs

    training_dataset = tf.data.TextLineDataset(file_name_train).map(parse).padded_batch(lm_batch_size, padded_shapes=([MAX_WORD_LENGTH], [MAX_WORD_LENGTH]))
    print('training:',training_dataset.output_shapes)
    validation_dataset = tf.data.TextLineDataset(file_name_validation).map(parse).padded_batch(lm_batch_size, padded_shapes=([MAX_WORD_LENGTH], [MAX_WORD_LENGTH]))
    test_dataset = tf.data.TextLineDataset(file_name_test).map(parse).batch(1)

    iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                              training_dataset.output_shapes)

    input_batch, output_batch = iterator.get_next()
    print('input_batch:',input_batch)
    #input_batch = tf.Print(input_batch,[input_batch],'input_batch : ',summarize=500)
    print('output_batch:',output_batch)
    #output_batch = tf.Print(output_batch,[output_batch],'output_batch : ',summarize=500)
    trining_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    test_init_op = iterator.make_initializer(test_dataset)


    # Input embedding mat
    print('input_embedding_mat:',input_embedding_mat)
    input_embedded = tf.nn.embedding_lookup(input_embedding_mat, input_batch)
    print('input_embedded:',input_embedded)
    non_zero_weights = tf.sign(input_batch)
    valid_words = tf.reduce_sum(non_zero_weights)
    # LSTM cell
    '''
    cell = tf.contrib.rnn.LSTMCell(num_hidden_units, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout_rate)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)

    # Output embedding
    output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                    [lm_vocab_size, num_hidden_units],
                                                    dtype=tf.float32)
    print('output_embedding_mat:',output_embedding_mat)
    output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                     [lm_vocab_size],
                                                     dtype=tf.float32)

    non_zero_weights = tf.sign(input_batch)
    valid_words = tf.reduce_sum(non_zero_weights)

    # Compute sequence length
    def get_length(non_zero_place):
        real_length = tf.reduce_sum(non_zero_place, 1)
        real_length = tf.cast(real_length, tf.int32)
        return real_length

    batch_length = get_length(non_zero_weights)


        # The shape of outputs is [lm_batch_size, max_length, num_hidden_units]
    outputs, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=input_embedded,
            sequence_length=batch_length,
            dtype=tf.float32
    )
    print('outputs:',outputs)
    def output_embedding(current_output):
        oputs = tf.add(
            tf.matmul(current_output, tf.transpose(output_embedding_mat)), output_embedding_bias)
        print('oputs:',tf.transpose(output_embedding_mat))
        return oputs

    # To compute the logits
    lm_logits = tf.map_fn(output_embedding, outputs)
    '''
    lm_logits = cal_gru_word(input_embedded, dropout_rate)
    # To compute the logits
    lm_logits = tf.map_fn(output_embedding, lm_logits)
    print('lm_logits:',lm_logits)
    lm_logits = tf.reshape(lm_logits, [-1, vocab_size])
    print('lm_logits:',lm_logits)
    lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(output_batch, [-1]), logits=lm_logits) #\
#           * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
    print('output_batch:',tf.reshape(output_batch, [-1]))
    # Train

    params = tf.trainable_variables()

    opt = tf.train.AdagradOptimizer(learning_rate)
    gradients = tf.gradients(lm_loss, params, colocate_gradients_with_ops=True)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
    #return updates
    

    lr = 2e-4
    h_representation = cal_gru_word(batch_embedded, keep_prob, reuse=True)
    batch_embedded_sent = cal_loss_logit_word(h_representation, keep_prob, reuse=True)
    logits, cl_loss = cal_loss_logit_sent(batch_embedded_sent, keep_prob, reuse=True)
    #print('cl_loss :',cl_loss)
    summary_cl_loss = tf.summary.scalar('cl_loss',cl_loss)
    summary_cl_vloss = tf.summary.scalar('cl_vloss',cl_loss)
    
    embedding_perturbated = add_perturbation(batch_embedded, cl_loss)

    h_representation_perturbated = cal_gru_word(embedding_perturbated, keep_prob, reuse=True)
    embedding_perturbated_sent = cal_loss_logit_word(h_representation_perturbated, keep_prob, reuse=True)
    ad_logits, ad_loss = cal_loss_logit_sent(embedding_perturbated_sent, keep_prob, reuse=True)
    #print('ad_loss :',ad_loss)
    summary_ad_loss = tf.summary.scalar('ad_loss',ad_loss)
    summary_ad_vloss = tf.summary.scalar('ad_vloss',ad_loss)

    sent_perturbated = add_perturbation(batch_embedded_sent, cl_loss)

    ad_logits_sent, ad_loss_sent = cal_loss_logit_sent(sent_perturbated, keep_prob, reuse=True)
    summary_ad_loss_sent = tf.summary.scalar('ad_loss_sent',ad_loss_sent)
    summary_ad_vloss_sent = tf.summary.scalar('ad_vloss_sent',ad_loss_sent)
    
    vat_loss = virtual_adversarial_loss(batch_embedded, logits, keep_prob, False)
    #print('vat_loss :',vat_loss)
    summary_vat_loss = tf.summary.scalar('vat_loss',vat_loss)
    summary_vat_vloss = tf.summary.scalar('vat_vloss',vat_loss)

    vat_loss_sent = virtual_adversarial_loss(batch_embedded_sent, logits, keep_prob, True)
    #print('vat_loss_sent :',vat_loss_sent)
    summary_vat_loss_sent = tf.summary.scalar('vat_loss_sent',vat_loss_sent)
    summary_vat_vloss_sent = tf.summary.scalar('vat_vloss_sent',vat_loss_sent)
    
    #cl_loss = tf.Print(cl_loss,[cl_loss],'cl_loss : ')
    #ad_loss = tf.Print(ad_loss,[ad_loss],'ad_loss : ')
    #ad_loss_sent = tf.Print(ad_loss_sent,[ad_loss_sent],'ad_loss_sent : ')
    #vat_loss = tf.Print(vat_loss,[vat_loss],'vat_loss : ')
    #vat_loss_sent = tf.Print(vat_loss_sent,[vat_loss_sent],'vat_loss_sent : ')
    loss = cl_loss
    loss += vat_loss
    loss += ad_loss
    loss += vat_loss_sent
    loss += ad_loss_sent
    summary_loss = tf.summary.scalar('loss',loss)
    summary_vloss = tf.summary.scalar('vloss',loss)
    for i, d in enumerate(GPU_device):
        with tf.device(d):  
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.90).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
            prediction = tf.argmax(tf.nn.softmax(logits), 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))
    summary_accuracy = tf.summary.scalar('accuracy',accuracy)
    summary_vaccuracy = tf.summary.scalar('vaccuracy',accuracy)
    train_op =tf.summary.merge([summary_cl_loss, summary_vat_loss, summary_ad_loss, summary_vat_loss_sent, summary_ad_loss_sent, summary_loss, summary_accuracy])
    #train_op =tf.summary.merge([summary_cl_loss, summary_loss, summary_ad_loss, summary_ad_loss_sent, summary_accuracy])
    val_op =tf.summary.merge([summary_cl_vloss, summary_vat_vloss, summary_ad_vloss, summary_vat_vloss_sent, summary_ad_vloss_sent, summary_vloss, summary_vaccuracy])
    #val_op =tf.summary.merge([summary_cl_vloss, summary_ad_loss, summary_ad_loss_sent, summary_vloss, summary_vaccuracy])

with tf.Session(graph=graph,config=config) as sess:
    #sess.run(tf.global_variables_initializer())
    print("Initialized! ")
    for i in tf.trainable_variables():
        print(i.name)
    #saver = tf.train.Saver({"input_embedding_mat": input_embedding_mat, "word_gru/rnn/gru_cell/gates/kernel": word_gru/rnn/gru_cell/gates/kernel, "word_gru/rnn/gru_cell/gates/bias":word_gru/rnn/gru_cell/gates/bias, "word_gru/rnn/gru_cell/candidate/kernel": word_gru/rnn/gru_cell/candidate/kernel, "word_gru/rnn/gru_cell/candidate/bias":word_gru/rnn/gru_cell/candidate/bias})
    saver = tf.train.Saver()

    '''
#----------------------------------------------------------------------------------------
    best_score = np.inf
    patience = 5
    epoch = 0
    VERBOSE = True

    # To indicate your test corpus
    test_file = "./gap_filling_exercise/gap_filling_exercise"

    #if not os.path.isfile("data/sg_vocab"):
    #    gen_vocab("ptb/sg_train")
    #if not os.path.isfile("data/sg_train.ids"):
    #    gen_id_seqs("ptb/sg_train")
    #    gen_id_seqs("ptb/sg_valid")


    while epoch < num_epochs:

        sess.run(trining_init_op, {file_name_train: "./data/SG_train.ids"})
        train_loss = 0.0
        train_valid_words = 0
        saver = tf.train.Saver()
        while True:

            try:
                _loss, _global_step, _valid_words, current_learning_rate, _ = sess.run(
                    [lm_loss, global_step, valid_words, learning_rate, updates],
                    {dropout_rate: 0.5})
                train_loss += np.sum(_loss)
                train_valid_words += _valid_words

                if _global_step % check_point_step == 0:

                    train_loss /= train_valid_words
                    print(train_loss)
                    train_ppl = math.exp(train_loss)
                    print "Training Step: {}, LR: {}".format(_global_step, current_learning_rate)
                    print "    Training PPL: {}".format(train_ppl)

                    train_loss = 0.0
                    train_valid_words = 0


            except tf.errors.OutOfRangeError:
                # The end of one epoch
                break


        sess.run(validation_init_op, {file_name_validation: "./data/SG_test.ids"})
        dev_loss = 0.0
        dev_valid_words = 0
        while True:
            try:
                _dev_loss, _dev_valid_words = sess.run(
                    [lm_loss, valid_words],
                    {dropout_rate: 1.0})

                dev_loss += np.sum(_dev_loss)
                dev_valid_words += _dev_valid_words
            except tf.errors.OutOfRangeError:
                dev_loss /= dev_valid_words
                dev_ppl = math.exp(dev_loss)
                print "Validation PPL: {}".format(dev_ppl)
                if dev_ppl < best_score:
                    patience = 1
                    saver.save(sess, "model/sogou_best_model.ckpt")
                    best_score = dev_ppl
                else:
                    patience -= 1

                if patience == 0:
                    epoch = num_epochs + 1

                break

#---------------------------------------------------------------------------------------
    '''
    saver.restore(sess, "model/sogou_best_model.ckpt")
    print("Start training")
    start = time.time()
    time_consumed = 0
    max_acc = 0
    model_name = 'han_adv_pretrain_sogou'
    hyperparameter = model_name + 'Embedding_'+ str(EMBEDDING_SIZE) +'_Hidden_'+ str(HIDDEN_SIZE) +'_Dropout_'+ str(KEEP_PROB) +'_Epsilon_adv_'+ str(epsilon) +'_Epsilon_vir_'+ str(epsilon2) +'_sent_'+ str(MAX_DOCUMENT_LENGTH) +'_word_'+ str(MAX_WORD_LENGTH) +'_lr_'+ str(lr)
    out_file = 'log_adv_vir_han_pretrain_sogou/'
    out_dir = out_file + hyperparameter +'/'
    if not os.path.exists(os.path.dirname(out_dir)):
        try:
            os.makedirs(os.path.dirname(out_dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    writer = tf.summary.FileWriter(out_dir,sess.graph)

    with open(out_file+ hyperparameter +'.txt', 'w') as log_file:
        for e in range(epochs):
            epoch_start = time.time()
            #print("Epoch %d start !" % (e + 1))
            num_batch = 0
            for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
                fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
                l, _, acc, train = sess.run([loss, optimizer, accuracy, train_op], feed_dict=fd)
                #print(num_batch,acc)
                num_batch+=1
                writer.add_summary(train)
            epoch_finish = time.time()
            num_val = 0
            total_acc = 0
            for x_batch, y_batch in fill_feed_dict(x_dev, y_dev, BATCH_SIZE):
                acc, _, val = sess.run([accuracy, loss, val_op], feed_dict={
                batch_x: x_batch,
                batch_y: y_batch,
                keep_prob: 1.0
                })
                #print(num_val,acc)
                total_acc +=acc
                num_val+=1    
                writer.add_summary(val)
            '''
            print("Validation accuracy: ", sess.run([accuracy, loss], feed_dict={
                batch_x: x_dev,
                batch_y: y_dev,
                keep_prob: 1.0
            }))
            '''
            print("Validation acc:",total_acc/num_val)
            if total_acc/num_val > max_acc:    
                max_acc = total_acc/num_val
            log_file.write(str(total_acc/num_val)+" ")

    print("Max Validation acc:",max_acc)

    print("Training finished, time consumed : ", time.time() - start, " s")
    #print("start predicting:  \n")
    #test_accuracy = sess.run([accuracy], feed_dict={batch_x: x_test, batch_y: y_test, keep_prob: 1})
    #print("Test accuracy : %f %%" % (test_accuracy[0] * 100))
