import tensorflow as tf
print(tf.__version__)
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
import sys, re
import nltk
from nltk import tokenize
from bs4 import BeautifulSoup
import collections
from collections import defaultdict
from utils.prepare_data import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.test.gpu_device_name())

# Hyperparameters
MAX_DOCUMENT_LENGTH = 25
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 128
BATCH_SIZE = 256
KEEP_PROB = 0.5
epsilon = 5.0  # IMDB ideal norm length
MAX_LABEL = 15
epochs = 10
INITIALIZER=tf.random_normal_initializer(stddev=0.1)
max_num_sentences = 3  # while batch_size = 64
max_sequence_length = 30
my_tensor_0 = tf.placeholder(tf.int32)
batch_size = tf.shape(my_tensor_0)[0]  # 
my_tensor_1 = tf.placeholder(tf.int32)
sequence_length = tf.shape(my_tensor_1)[0]  # 
my_tensor_2 = tf.placeholder(tf.int32)
num_sentences = tf.shape(my_tensor_2)[0]

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[.]", " . ", string)    
    string = re.sub(r'[^\w.]', ' ', string)
    return string.strip().lower()

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  print("building dataset")
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  count = dict(count)
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
  #return dictionary#, reversed_dictionary

vocab = []
def process_yelp(data_yelp,is_train):
    global vocab
    text_train = []
    text_dev = []
    text_test = []
    label_train = []
    label_dev = []
    for idx in range(len(data_yelp)):
        sys.stdout.write("\rProcessing ---- %d"%idx)
        sys.stdout.flush()
        if pd.isnull(data_yelp[idx]):
    	    data_train.drop(data_yelp.index[idx])
    	    print("Empty row:%d"%idx)
        else :
            #parse the sentences into beautifulsoup object
            text = BeautifulSoup(data_yelp[idx])
            text = clean_str(text.get_text().encode('ascii','ignore'))
            text_to_tf = text.split()
            #insert clear text into texts array
            #texts.append(text)
            #Return a sentence-tokenized copy of text( divide string into substring by punkt)
            sentences = tokenize.sent_tokenize(text)
            if is_train:
                text_train.append(text_to_tf)
            else:
                text_dev.append(text_to_tf)

            for word in text_to_tf:
                vocab.append(word) 

    if not is_train:
        data, count, dictionary, reversed_dictionary = build_dataset(vocab,50000)
        print("1st vocab loaded")
        return dictionary, count

def sentence_padding(sentence, max_length, token):
    """
    :param sentence: a list
    :param num:  num token2id["UNK"]
    :return:
    """
    temp_list = []
    if len(sentence) <= max_length:
        for _ in range (max_length-len(sentence)):
            temp_list.append(token["UNK"])
        temp_list.extend(sentence)
    else:
        temp_list = sentence[max_length*(-1):]
    #print(temp_list)
    return temp_list

def normarlized_input(sentences, token2id):
    total_train_data_size = len(sentences)
    raw_input_x = []
    for sentence in sentences:
        #print(sentence)
        temp_list_1 = []
        for idx in range(len(sentence)):
            #print(sentence[idx].strip())
            if sentence[idx].strip() == ".":
                temp_list_1.append(idx)
        if len(temp_list_1) > max_num_sentences:
            begin = temp_list_1[(max_num_sentences + 1) * -1] + 1
            short_sentence = sentence[begin:]
        else:
            short_sentence = sentence
        #print('short_sentence:',short_sentence)
        # temp_list = [token2id[k] for k in short_sentence]  # num_sentences 
        temp_list = []
        for k in short_sentence:
            if k in token2id:
                temp_list.append(token2id[k])
                #print('word',k)
                #print('token',token2id[k])
            else:
                temp_list.append(token2id["UNK"])
                # temp_list.append(-1)
        temp_list_1 = []
        for idx in range(len(short_sentence)):
            if short_sentence[idx].strip() == ".":
                temp_list_1.append(idx)
        document = []
        for i in range(len(temp_list_1)):  # num_sentences
            if i == 0:
                #print(temp_list[:temp_list_1[i] + 1])
                document.append(sentence_padding(sentence=temp_list[:temp_list_1[i] + 1],max_length=max_sequence_length,token=token2id))
            else:
                #print(temp_list[temp_list_1[i - 1] + 1:temp_list_1[i] + 1])
                document.append(sentence_padding(sentence=temp_list[temp_list_1[i - 1] + 1:temp_list_1[i] + 1],max_length=max_sequence_length,token=token2id))
        if len(document) < max_num_sentences:
            pad_list = [token2id["UNK"] for _ in range(max_sequence_length)]
            for _ in range(max_num_sentences - len(document)):
                document.insert(0, pad_list)
        # len(document) = num_sentences, len(element) = sequence_length
        # document: [num_sentences, sequence_length]
        raw_input_x.append(document)
        #print(document)
    return raw_input_x

def load_embedding_from_disks(glove_filename, dictionary, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct 
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    
    with open(glove_filename, 'r') as glove_file:
        j = 0
        for (i, line) in enumerate(glove_file):
            
            split = line.split(' ')
            
            word = split[0]
            if word in dictionary:
                if j<100:
                    print(word)
                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )
            
                if with_indexes:
                    word_to_index_dict[word] = j
                    index_to_embedding_array.append(representation)
                else:
                    word_to_embedding_dict[word] = representation
                j+=1
    print("j",j)
    word_to_index_dict['UNK'] = j
    index_to_embedding_array.append(np.random.rand(200,))
    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = j + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        #index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        index_to_embedding_array = np.array(index_to_embedding_array)
        return word_to_index_dict, index_to_embedding_array, _LAST_INDEX+1
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict, j

def get_freq(vocab_freq, word2idx,dictionary):
    """get a frequency dict format as {word_idx: word_freq}"""
    words = dictionary.keys()
    vocab_size = len(word2idx) +1
    print(len(word2idx))
    freq = [0] * vocab_size
    for word in words:
        word_idx = word2idx[word]
        #print(word_idx)
        word_freq = vocab_freq[word]
        freq[word_idx] = word_freq
    return freq

#dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia', test_with_fake_data=False)
#print(dbpedia)

# load data
x_train, y_train = load_data("../dbpedia_data/dbpedia_csv/train.csv", sample_ratio=0.01)
print(x_train)
print(y_train)
x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/test.csv", sample_ratio=0.1)

x_train = x_train.reset_index()
x_train = x_train.drop('index', 1)['content']
print(x_train)
x_test = x_test.reset_index()
x_test = x_test.drop('index', 1)['content']

process_yelp(x_train,True)
dictionary_yelp,vocab_freq = process_yelp(x_test,False)
#print(vocab_freq)
word_to_index, index_to_embedding, vocab_size = load_embedding_from_disks('../../nextlabs/Glove/glove.6B.200d.txt', dictionary_yelp, with_indexes=True)
word2idx = word_to_index
x_train = normarlized_input(x_train,word2idx)
#print(x_train)

#v_freq = get_freq(vocab_freq, token2id,dictionary_yelp)
#print(v_freq)
# data preprocessing
#x_train, x_test, vocab_freq, word2idx,vocab_size = \
#    data_preprocessing_with_dict(x_train, x_test, MAX_DOCUMENT_LENGTH)
#print(x_train)
#print(x_test)
print("Vocab size: ", vocab_size)

# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
print("Validation size: ", dev_size)

#x_test, y_test = load_data("../dbpedia_data/dbpedia_csv/testgg.csv", sample_ratio=1)



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
    print("Weights: ", weights)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev


graph = tf.Graph()
with graph.as_default():

    def __init__(self):
        self.instantiate_weights()

    def instantiate_weights():
        """define all weights here"""
        print("Loading embedding from disks...")
        print("Embedding loaded from disks.")

        batch_size = 128  # Any size is accepted

        with tf.name_scope("embedding_projection"):
            W_projection = tf.get_variable("W_projection", shape=[HIDDEN_SIZE * 2, MAX_LABEL],
                                                initializer=INITIALIZER)  # [embed_size,label_size]
           # self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                                #initializer=self.initializer) 
            with tf.device('/cpu:0'):
                Embedding = tf.get_variable("Embedding", shape=index_to_embedding.shape,
                                                initializer=None)
            b_projection = tf.get_variable("b_projection", shape=[MAX_LABEL])

        with tf.name_scope("attention"):
            W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                      shape=[HIDDEN_SIZE * 2, HIDDEN_SIZE * 2],
                                                      initializer=INITIALIZER)
            W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[HIDDEN_SIZE * 2])

            W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",
                                                          shape=[HIDDEN_SIZE * 2, HIDDEN_SIZE * 2],
                                                          initializer=INITIALIZER)
            W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[HIDDEN_SIZE * 2])
            context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[HIDDEN_SIZE * 2],
                                                        initializer=INITIALIZER)
            context_vecotor_sentence = tf.get_variable("what_is_the_informative_sentence",
                                                            shape=[HIDDEN_SIZE * 2], initializer=INITIALIZER)
        return W_projection, b_projection, W_w_attention_word, W_b_attention_word, W_w_attention_sentence, W_b_attention_sentence, context_vecotor_word, context_vecotor_sentence

    W_projection, b_projection, W_w_attention_word, W_b_attention_word, W_w_attention_sentence, W_b_attention_sentence, context_vecotor_word, context_vecotor_sentence = instantiate_weights()
    batch_x = tf.placeholder(tf.int32, [None, None, None], name="input_x")
    batch_y = tf.placeholder(tf.float32, [None,])
    keep_prob = tf.placeholder(tf.float32)
    vocab_freqs = tf.constant(get_freq(vocab_freq, word2idx,dictionary_yelp), dtype=tf.float32, shape=(vocab_size, 1))

    weights = vocab_freqs / tf.reduce_sum(vocab_freqs)

    embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))
    W_fc = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, MAX_LABEL], stddev=0.1))
    b_fc = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))

    embedding_norm = normalize(embeddings_var, weights)
    batch_embedded = tf.nn.embedding_lookup(embedding_norm, batch_x)


    def gru_forward_word_level(embedded_words):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:[batch_size*num_sentences,sentence_length,hidden_size]
        """
        with tf.variable_scope("gru_weights_word_level_forward"):
            with tf.device("/gpu:0"):
                wf_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                init_state = wf_cell.zero_state(batch_size=BATCH_SIZE*MAX_DOCUMENT_LENGTH, dtype=tf.float32)  # [batch_size, hidden_size]
                output, state = tf.nn.dynamic_rnn(wf_cell, embedded_words, initial_state=init_state, time_major=False)
        # output: [batch_size*num_sentences,sentence_length,hidden_size]
        # state:  [batch_size*num_sentences, hidden_size]


        # output_splitted = tf.split(output, self.sequence_length,
        #                                    axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,hidden_size]
        # output_squeeze = [tf.squeeze(x, axis=1) for x in
        #                           output_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]

        return output

    def gru_backward_word_level(embedded_words):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: [batch_size*num_sentences,sentence_length,hidden_size]
        """
        embedded_words_reverse = tf.reverse(embedded_words, [0])
        with tf.variable_scope("gru_weights_word_level_backward"):
            with tf.device("/gpu:0"):
                wb_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                init_state = wb_cell.zero_state(batch_size=BATCH_SIZE*MAX_DOCUMENT_LENGTH, dtype=tf.float32)  # [batch_size, hidden_size]
                output, state = tf.nn.dynamic_rnn(wb_cell, embedded_words_reverse, initial_state=init_state, time_major=False)
        # output: [batch_size*num_sentences,sentence_length,hidden_size]
        # state:  [batch_size*num_sentences, hidden_size]

        # output_splitted = tf.split(output, self.sequence_length,
        #                            axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,hidden_size]
        # output_squeeze = [tf.squeeze(x, axis=1) for x in
        #                   output_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # output_squeeze.reverse()
                output = tf.reverse(output, [2])
        return output

    def gru_forward_sentence_level(sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:[batch_size,num_sentences,hidden_size]
        """
        with tf.variable_scope("gru_weights_sentence_level_forward"):
            with tf.device("/gpu:0"):
                sf_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                init_state = sf_cell.zero_state(batch_size=BATCH_SIZE,
                                         dtype=tf.float32)  # [batch_size, hidden_size]
                output, state = tf.nn.dynamic_rnn(sf_cell, sentence_representation, initial_state=init_state, time_major=False)
        return output

    def gru_backward_sentence_level(sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:[batch_size,num_sentences,hidden_size]
        """
        sentence_representation_reverse = tf.reverse(sentence_representation, [2])
        with tf.variable_scope("gru_weights_sentence_level_backward"):
            with tf.device("/gpu:0"):
                sb_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                init_state = sb_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)  # [batch_size, hidden_size]
                output, state = tf.nn.dynamic_rnn(sb_cell, sentence_representation_reverse, initial_state=init_state, time_major=False)
                output = tf.reverse(output, [2])
        return output

    def attention_word_level(hidden_state, W_w_attention_word, W_b_attention_word, context_vecotor_word):
        """
        input:[batch_size*num_sentences,sentence_length,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        with tf.device("/GPU:0"):
            hidden_state_2 = tf.reshape(hidden_state, shape=[-1,
                                                          HIDDEN_SIZE * 2])
            hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     W_w_attention_word) + W_b_attention_word)
            hidden_representation = tf.reshape(hidden_representation, shape=[-1, max_sequence_length,
                                                                         HIDDEN_SIZE * 2])
            hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       context_vecotor_word)
            attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # [batch_size*num_sentences,sentence_length]
            attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # [batch_size*num_sentences,1]
            p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # [batch_size*num_sentences,sentence_length]
            p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # [batch_size*num_sentences,sentence_length,1]
            sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state)  # [batch_size*num_sentences,sentence_length, hidden_size*2]
            sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)  # [batch_size*num_sentences, hidden_size*2]
            return sentence_representation

    def attention_sentence_level(hidden_state_sentence, W_w_attention_sentence, W_b_attention_sentence, context_vecotor_sentence):
        """
        input: [batch_size,num_sentences,hidden_size]
        :return:representation.shape:[batch_size,hidden_size*2]
        """
        with tf.device("/GPU:0"):
            hidden_state_2 = tf.reshape(hidden_state_sentence,
                                    shape=[-1, HIDDEN_SIZE * 2])
            hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     W_w_attention_sentence) + W_b_attention_sentence)
            hidden_representation = tf.reshape(hidden_representation, shape=[-1, max_num_sentences,
                                                                         HIDDEN_SIZE * 2])

            hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       context_vecotor_sentence)
            attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)
            attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)

            p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
            p_attention_expanded = tf.expand_dims(p_attention, axis=2)
            sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_sentence)
            sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
            return sentence_representation

    def cal_loss_logit(batch_embedded, keep_prob, reuse=True, scope="loss"):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            batch_embedded_reshaped = tf.reshape(batch_embedded, shape=[-1, max_sequence_length, EMBEDDING_SIZE])
            hidden_state_forward = gru_forward_word_level(batch_embedded_reshaped)
            hidden_state_backward = gru_backward_word_level(batch_embedded_reshaped)
            hidden_state = tf.concat([hidden_state_forward, hidden_state_backward], axis=2)
            #rnn_outputs, _ = bi_rnn(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE),
            #                        inputs=batch_embedded_reshaped, dtype=tf.float32)
            #hidden_state_forward = gru_forward_word_level(batch_embedded)
            #hidden_state_backward = gru_backward_word_level(batch_embedded)
            #hidden_state = tf.concat([hidden_state_forward, hidden_state_backward], axis=2)
            # Attention
            #H = tf.add(rnn_outputs[0], rnn_outputs[1])  # fw + bw
            sentence_representation = attention_word_level(hidden_state, W_w_attention_word, W_b_attention_word, context_vecotor_word)  # output
            sentence_representation = tf.reshape(sentence_representation, shape=[-1, num_sentences, HIDDEN_SIZE * 2])
            #rnn_outputs_sent, _ = bi_rnn(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE),
            #                        inputs=sentence_representation, dtype=tf.float32)
            #hidden_state_forward = gru_forward_word_level(batch_embedded)
            #hidden_state_backward = gru_backward_word_level(batch_embedded)
            #hidden_state = tf.concat([hidden_state_forward, hidden_state_backward], axis=2)
            # Attention
            #H_sent = tf.add(rnn_outputs_sent[0], rnn_outputs_sent[1])
            hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)
            hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)
            hidden_state_sentence = tf.concat([hidden_state_forward_sentences, hidden_state_backward_sentences], axis=2)
            document_representation = attention_sentence_level(hidden_state_sentence, W_w_attention_sentence, W_b_attention_sentence, context_vecotor_sentence)
            #M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
            # alpha (bs * sl, 1)
            #alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
            #r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_DOCUMENT_LENGTH,
            #                                                             1]))  # supposed to be (batch_size * HIDDEN_SIZE, 1)
            #r = tf.squeeze(r)
            #h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE
            # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
            drop = tf.nn.dropout(document_representation, keep_prob)

            # Fully connected layer dense layer
            y_hat = tf.nn.xw_plus_b(drop, W_fc, b_fc)
	#v2
        return y_hat, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=batch_y))


    lr = 1e-3
    logits, cl_loss = cal_loss_logit(batch_embedded, keep_prob, reuse=tf.AUTO_REUSE)
    embedding_perturbated = add_perturbation(batch_embedded, cl_loss)
    ad_logits, ad_loss = cal_loss_logit(embedding_perturbated, keep_prob, reuse=tf.AUTO_REUSE)
    loss = cl_loss + ad_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(logits), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized! ")

    print("Start training")
    start = time.time()
    time_consumed = 0
    for e in range(epochs):

        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        epoch_finish = time.time()
        print("Validation accuracy: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        }))

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("start predicting:  \n")
    test_accuracy = sess.run([accuracy], feed_dict={batch_x: x_test, batch_y: y_test, keep_prob: 1})
    print("Test accuracy : %f %%" % (test_accuracy[0] * 100))
