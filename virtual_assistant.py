import itertools
import logging
import os
import tarfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def process_models():
    # def download_models():
    #     from neural.filedownload import url_download
    #     if not os.path.isfile(glove_zip_file) and not os.path.isfile(glove_vectors_file):
    #         url_download("http://nlp.stanford.edu/data/glove.6B.zip", glove_zip_file)
    #     if not os.path.isfile(data_set_zip) and not (os.path.isfile(train_set_file) and os.path.isfile(test_set_file)):
    #         url_download("https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz", data_set_zip)

    def unzip_single_file(zip_file_name, output_file_name):
        """
            If the output file is already created, don't recreate
            If the output file does not exist, create it from the zipFile
        """
        if not os.path.isfile(output_file_name):
            with open(output_file_name, 'wb') as out_file:
                with zipfile.ZipFile(zip_file_name) as zipped:
                    for info in zipped.infolist():
                        if output_file_name in info.filename:
                            with zipped.open(info) as requested_file:
                                out_file.write(requested_file.read())
                                return

    def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
        if not os.path.isfile(output_file_name):
            with tarfile.open(zip_file_name) as un_zipped:
                un_zipped.extract(interior_relative_path + output_file_name)

    # download_models()
    unzip_single_file(glove_zip_file, glove_vectors_file)
    targz_unzip_single_file(data_set_zip, train_set_file, path + "tasks_1-20_v1-2/en/")
    targz_unzip_single_file(data_set_zip, test_set_file, path + "tasks_1-20_v1-2/en/")


logging.basicConfig(level=logging.DEBUG)
glove_zip_file = "glove.6B.zip"
path = "./neural/"
glove_vectors_file = "glove.6B.50d.txt"
# 15 MB
data_set_zip = "tasks_1-20_v1-2.tar.gz"
# Select "task 5"
train_set_file = "qa5_three-arg-relations_train.txt"
test_set_file = "qa5_three-arg-relations_test.txt"
train_set_post_file = path + "tasks_1-20_v1-2/en/" + train_set_file
test_set_post_file = path + "tasks_1-20_v1-2/en/" + test_set_file

# process_models()

'''-----------------------------------------------------------'''
# Deserialize GloVe vectors
# print(os.getcwd())
# os.chdir("..")
# print(os.getcwd())
from entity.embeddingrepo import EmbeddingDbRepo

glove_wordmap = {}
embrepo = EmbeddingDbRepo()
gloves = embrepo.get()
for glove in gloves:
    name, vector = list(glove)[1], list(glove)[2]
    glove_wordmap[name] = np.fromstring(vector, sep=" ")
# glove_wordmap = {}
# with open(glove_vectors_file, "r", encoding="utf8") as glove:
#     for line in glove:
#         name, vector = tuple(line.split(" ", 1))
#         glove_wordmap[name] = np.fromstring(vector, sep=" ")
wvecs = []
for item in glove_wordmap.items():
    wvecs.append(item[1])
s = np.vstack(wvecs)

# Gather the distribution hyperparameters
v = np.var(s, 0)
m = np.mean(s, 0)
RS = np.random.RandomState()


# print(os.getcwd())
def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk] = RS.multivariate_normal(m, np.diag(v))
    return glove_wordmap[unk]


'''-----------------------------------------------------------'''


def sentence2sequence(sentence):
    tokens = sentence.strip('"(),-').lower().split(" ")
    rows = []
    words = []
    # Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0:
            word = token[:i]
            # print(word)
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
                continue
            else:
                i = i - 1
            if i == 0:
                # word OOV
                # https://arxiv.org/pdf/1611.01436.pdf
                rows.append(fill_unk(token))
                words.append(token)
                break
    return np.array(rows), words


'''-----------------------------------------------------------'''


def contextualize(category, val=False):
    data = []
    context = []
    if val:
        from entity.patternrepo import PatternDbRepo
        patrepo = PatternDbRepo()
        pattern = patrepo.all(category)
        for p in pattern:
            # print(p.episodes)
            p.category_id = category
            context = []
            # print("new", "-------------------------------------------")
            for pkey, pval in p.episodes.items():
                # print(ep)
                context.append(sentence2sequence(pval))

                # print(context)

                if pkey in p.questions:
                    data.append((tuple(zip(*context)) +
                                 sentence2sequence(p.questions[pkey][0]) +
                                 sentence2sequence(p.questions[pkey][1]) +
                                 ([int(pkey)],)))
            # print("-------------------------------------------")



    else:
        with open(category, "r", encoding="utf8") as train:
            for line in train:
                l, ine = tuple(line.split(" ", 1))
                # Split the line numbers from the sentences they refer to.
                if l is "1":
                    # New contexts always start with 1,
                    # so this is a signal to reset the context.
                    context = []
                if "\t" in ine:
                    # Tabs are the separator between questions and answers,
                    # and are not present in context statements.
                    # print(tuple(ine.split("\t")))
                    question, answer, support = tuple(ine.split("\t"))

                    # print("old", question, answer, support.replace("\n",''))
                    # print("old", "-------------------------------------------")
                    # print(context)
                    # print("-------------------------------------------")
                    data.append((tuple(zip(*context)) +
                                 sentence2sequence(question) +
                                 sentence2sequence(answer) +
                                 ([int(s) for s in support.replace("\n", '')],)))
                    # Multiple questions may refer to the same context, so we don't reset it.
                else:
                    # Context sentence.
                    # print(ine.replace("\n", ''))
                    context.append(sentence2sequence(ine.replace("\n", '')))
                # print("-------------------------------------------")

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(data[0])
    # print(data)
    return data


train_set_post_file = "./neural/tasks_1-20_v1-2/en/" + "qa5.txt"
# train_data = contextualize(train_set_post_file)

train_data = contextualize(1, val=True)
print("finish")
print(os.getcwd())
train_set_post_file = "E:/Desktop/Licenta/master/virtual-client-assistant/neural/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt"
test_data = contextualize(train_set_post_file)

'''-----------------------------------------------------------'''
final_train_data = []


def finalize(data):
    """
    Prepares data generated by contextualize() for use in the network.
    """
    final_data = []
    for cqas in data:
        contextvs, contextws, qvs, qws, avs, aws, spt = cqas

        lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
        context_vec = np.concatenate(contextvs)
        context_words = sum(contextws, [])

        # Location markers for the beginnings of new sentences.
        sentence_ends = np.array(list(lengths))
        final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws))
    return np.array(final_data)


final_train_data = finalize(train_data)
final_test_data = finalize(test_data)
'''-----------------------------------------------------------'''
tf.reset_default_graph()
'''-----------------------------------------------------------'''
# Hyperparameters

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 128

# The number of dimensions in our word vectorizations.
D = 50

# How quickly the network learns. Too high, and we may run into numeric instability
# or other issues.
learning_rate = 0.005

# Dropout probabilities. For a description of dropout and what these probabilities are,
# see Entailment with TensorFlow.
input_p, output_p = 0.5, 0.5

# How many questions we train on at a time.
batch_size = 128

# Number of passes in episodic memory. We'll get to this later.
passes = 4

# Feed Forward layer sizes: the number of dimensions used to store data passed from feed-forward layers.
ff_hidden_size = 256

weight_decay = 0.00000001
# The strength of our regularization. Increase to encourage sparsity in episodic memory,
# but makes training slower. Don't make this larger than leraning_rate.

training_iterations_count = 400000
# How many questions the network trains on each time it is trained.
# Some questions are counted multiple times.

display_step = 100
# How many iterations of training occur before each validation check.
'''-----------------------------------------------------------'''
# Input Module

# Context: A [batch_size, maximum_context_length, word_vectorization_dimensions] tensor
# that contains all the context information.
context = tf.placeholder(tf.float32, [None, None, D], "context")
context_placeholder = context  # I use context as a variable name later on

# input_sentence_endings: A [batch_size, maximum_sentence_count, 2] tensor that
# contains the locations of the ends of sentences.
input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

# recurrent_cell_size: the number of hidden units in recurrent layers.
input_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

# input_p: The probability of maintaining a specific hidden input unit.
# Likewise, output_p is the probability of maintaining a specific hidden output unit.
gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, input_p, output_p)

# dynamic_rnn also returns the final internal state. We don't need that, and can
# ignore the corresponding output (_).
input_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, context, dtype=tf.float32, scope="input_module")

# cs: the facts gathered from the context.
cs = tf.gather_nd(input_module_outputs, input_sentence_endings)
# to use every word as a fact, useful for tasks with one-sentence contexts
s = input_module_outputs
'''-----------------------------------------------------------'''
# Question Module

# query: A [batch_size, maximum_question_length, word_vectorization_dimensions] tensor
#  that contains all of the questions.

query = tf.placeholder(tf.float32, [None, None, D], "query")

# input_query_lengths: A [batch_size, 2] tensor that contains question length information.
# input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range()
# so that it plays nice with gather_nd.
input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, query, dtype=tf.float32,
                                               scope=tf.VariableScope(True, "input_module"))

# q: the question states. A [batch_size, recurrent_cell_size] tensor.
q = tf.gather_nd(question_module_outputs, input_query_lengths)
'''-----------------------------------------------------------'''
# Episodic Memory

# make sure the current memory (i.e. the question vector) is broadcasted along the facts dimension
size = tf.stack([tf.constant(1), tf.shape(cs)[1], tf.constant(1)])
re_q = tf.tile(tf.reshape(q, [-1, 1, recurrent_cell_size]), size)

# Final output for attention, needs to be 1 in order to create a mask
output_size = 1

# Weights and biases
attend_init = tf.random_normal_initializer(stddev=0.1)
w_1 = tf.get_variable("attend_w1", [1, recurrent_cell_size * 7, recurrent_cell_size],
                      tf.float32, initializer=attend_init)
w_2 = tf.get_variable("attend_w2", [1, recurrent_cell_size, output_size],
                      tf.float32, initializer=attend_init)

b_1 = tf.get_variable("attend_b1", [1, recurrent_cell_size],
                      tf.float32, initializer=attend_init)
b_2 = tf.get_variable("attend_b2", [1, output_size],
                      tf.float32, initializer=attend_init)

# Regulate all the weights and biases
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_1))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_1))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_2))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_2))


def attention(c, mem, existing_facts):
    """
    Custom attention mechanism.
    c: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor
        that contains all the facts from the contexts.
    mem: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor that
        contains the current memory. It should be the same memory for all facts for accurate results.
    existing_facts: A [batch_size, maximum_sentence_count, 1] tensor that
        acts as a binary mask for which facts exist and which do not.

    """
    with tf.variable_scope("attending") as scope:
        # attending: The metrics by which we decide what to attend to.
        attending = tf.concat([c, mem, re_q, c * re_q, c * mem, (c - re_q) ** 2, (c - mem) ** 2], 2)

        # m1: First layer of multiplied weights for the feed-forward network.
        #     We tile the weights in order to manually broadcast, since tf.matmul does not
        #     automatically broadcast batch matrix multiplication as of TensorFlow 1.2.
        m1 = tf.matmul(attending * existing_facts,
                       tf.tile(w_1, tf.stack([tf.shape(attending)[0], 1, 1]))) * existing_facts
        # bias_1: A masked version of the first feed-forward layer's bias
        #     over only existing facts.

        bias_1 = b_1 * existing_facts

        # tnhan: First nonlinearity. In the original paper, this is a tanh nonlinearity;
        #        choosing relu was a design choice intended to avoid issues with
        #        low gradient magnitude when the tanh returned values close to 1 or -1.
        tnhan = tf.nn.relu(m1 + bias_1)

        # m2: Second layer of multiplied weights for the feed-forward network.
        #     Still tiling weights for the same reason described in m1's comments.
        m2 = tf.matmul(tnhan, tf.tile(w_2, tf.stack([tf.shape(attending)[0], 1, 1])))

        # bias_2: A masked version of the second feed-forward layer's bias.
        bias_2 = b_2 * existing_facts

        # norm_m2: A normalized version of the second layer of weights, which is used
        #     to help make sure the softmax nonlinearity doesn't saturate.
        norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

        # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor.
        #     We make norm_m2 a sparse tensor, then make it dense again after the operation.
        softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:, :-1]
        softmax_gather = tf.gather_nd(norm_m2[..., 0], softmax_idx)
        softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
        softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)), -1)


# facts_0s: a [batch_size, max_facts_length, 1] tensor
#     whose values are 1 if the corresponding fact exists and 0 if not.
facts_0s = tf.cast(tf.count_nonzero(input_sentence_endings[:, :, -1:], -1, keepdims=True), tf.float32)

with tf.variable_scope("Episodes") as scope:
    attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

    # memory: A list of all tensors that are the (current or past) memory state
    #   of the attention mechanism.
    memory = [q]

    # attends: A list of all tensors that represent what the network attends to.
    attends = []
    for a in range(passes):
        # attention mask
        attend_to = attention(cs, tf.tile(tf.reshape(memory[-1], [-1, 1, recurrent_cell_size]), size),
                              facts_0s)

        # Inverse attention mask, for what's retained in the state.
        retain = 1 - attend_to

        # GRU pass over the facts, according to the attention mask.
        while_valid_index = (lambda state, index: index < tf.shape(cs)[1])
        update_state = (lambda state, index: (attend_to[:, index, :] *
                                              attention_gru(cs[:, index, :], state)[0] +
                                              retain[:, index, :] * state))
        # start loop with most recent memory and at the first index
        memory.append(tuple(tf.while_loop(while_valid_index,
                                          (lambda state, index: (update_state(state, index), index + 1)),
                                          loop_vars=[memory[-1], 0]))[0])

        attends.append(attend_to)

        # Reuse variables so the GRU pass uses the same variables every pass.
        scope.reuse_variables()
'''-----------------------------------------------------------'''
# Answer Module

# a0: Final memory state. (Input to answer module)
a0 = tf.concat([memory[-1], q], -1)

# fc_init: Initializer for the final fully connected layer's weights.
fc_init = tf.random_normal_initializer(stddev=0.1)

with tf.variable_scope("answer"):
    # w_answer: The final fully connected layer's weights.
    w_answer = tf.get_variable("weight", [recurrent_cell_size * 2, D],
                               tf.float32, initializer=fc_init)
    # Regulate the fully connected layer's weights
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                         tf.nn.l2_loss(w_answer))

    # The regressed word. This isn't an actual word yet;
    #    we still have to find the closest match.
    logit = tf.expand_dims(tf.matmul(a0, w_answer), 1)

    # Make a mask over which words exist.
    with tf.variable_scope("ending"):
        all_ends = tf.reshape(input_sentence_endings, [-1, 2])
        range_ends = tf.range(tf.shape(all_ends)[0])
        ends_indices = tf.stack([all_ends[:, 0], range_ends], axis=1)
        ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:, 1],
                                          [tf.shape(q)[0], tf.shape(all_ends)[0]]),
                            axis=-1)
        range_ind = tf.range(tf.shape(ind)[0])
        mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1),
                                          tf.ones_like(range_ind), [tf.reduce_max(ind) + 1,
                                                                    tf.shape(ind)[0]]), bool)
        # A bit of a trick. With the locations of the ends of the mask (the last periods in
        #  each of the contexts) as 1 and the rest as 0, we can scan with exclusive or
        #  (starting from all 1). For each context in the batch, this will result in 1s
        #  up until the marker (the location of that last period) and 0s afterwards.
        mask = tf.scan(tf.logical_xor, mask_ends, tf.ones_like(range_ind, dtype=bool))

    # We score each possible word inversely with their Euclidean distance to the regressed word.
    #  The highest score (lowest distance) will correspond to the selected word.
    logits = -tf.reduce_sum(tf.square(context * tf.transpose(tf.expand_dims(
        tf.cast(mask, tf.float32), -1), [1, 0, 2]) - logit), axis=-1)
'''-----------------------------------------------------------'''
# Training

# gold_standard: The real answers.
gold_standard = tf.placeholder(tf.float32, [None, 1, D], "answer")
with tf.variable_scope('accuracy'):
    eq = tf.equal(context, gold_standard)
    corrbool = tf.reduce_all(eq, -1)
    logloc = tf.reduce_max(logits, -1, keepdims=True)
    # locs: A boolean tensor that indicates where the score
    #  matches the minimum score. This happens on multiple dimensions,
    #  so in the off chance there's one or two indexes that match
    #  we make sure it matches in all indexes.
    locs = tf.equal(logits, logloc)

    # correctsbool: A boolean tensor that indicates for which
    #   words in the context the score always matches the minimum score.
    correctsbool = tf.reduce_any(tf.logical_and(locs, corrbool), -1)
    # corrects: A tensor that is simply correctsbool cast to floats.
    corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32),
                        tf.zeros_like(correctsbool, dtype=tf.float32))

    # corr: corrects, but for the right answer instead of our selected answer.
    corr = tf.where(corrbool, tf.ones_like(corrbool, dtype=tf.float32),
                    tf.zeros_like(corrbool, dtype=tf.float32))
with tf.variable_scope("loss"):
    # Use sigmoid cross entropy as the base loss,
    #  with our distances as the relative probabilities. There are
    #  multiple correct labels, for each location of the answer word within the context.
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.nn.l2_normalize(logits, -1),
                                                   labels=corr)

    # Add regularization losses, weighted by weight_decay.
    total_loss = tf.reduce_mean(loss) + weight_decay * tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

# TensorFlow's default implementation of the Adam optimizer works. We can adjust more than
#  just the learning rate, but it's not necessary to find a very good optimum.
optimizer = tf.train.AdamOptimizer(learning_rate)

# Once we have an optimizer, we ask it to minimize the loss
#   in order to work towards the proper training.
opt_op = optimizer.minimize(total_loss)
'''-----------------------------------------------------------'''
# Initialize variables
init = tf.global_variables_initializer()

# Launch the TensorFlow session
sess = tf.Session()
sess.run(init)
'''-----------------------------------------------------------'''


def prep_batch(batch_data, more_data=False):
    """
        Prepare all the preproccessing that needs to be done on a batch-by-batch basis.
    """
    context_vec, sentence_ends, questionvs, spt, context_words, cqas, answervs, _ = zip(*batch_data)
    ends = list(sentence_ends)
    maxend = max(map(len, ends))
    aends = np.zeros((len(ends), maxend))
    for index, i in enumerate(ends):
        for indexj, x in enumerate(i):
            aends[index, indexj] = x - 1
    new_ends = np.zeros(aends.shape + (2,))

    for index, x in np.ndenumerate(aends):
        new_ends[index + (0,)] = index[0]
        new_ends[index + (1,)] = x

    contexts = list(context_vec)
    max_context_length = max([len(x) for x in contexts])
    contextsize = list(np.array(contexts[0]).shape)
    contextsize[0] = max_context_length
    final_contexts = np.zeros([len(contexts)] + contextsize)

    contexts = [np.array(x) for x in contexts]
    for i, context in enumerate(contexts):
        final_contexts[i, 0:len(context), :] = context
    max_query_length = max(len(x) for x in questionvs)
    querysize = list(np.array(questionvs[0]).shape)
    querysize[:1] = [len(questionvs), max_query_length]
    queries = np.zeros(querysize)
    querylengths = np.array(list(zip(range(len(questionvs)), [len(q) - 1 for q in questionvs])))
    questions = [np.array(q) for q in questionvs]
    for i, question in enumerate(questions):
        queries[i, 0:len(question), :] = question
    data = {context_placeholder: final_contexts, input_sentence_endings: new_ends,
            query: queries, input_query_lengths: querylengths, gold_standard: answervs}
    return (data, context_words, cqas) if more_data else data


'''-----------------------------------------------------------'''
# Prepare validation set
print(final_test_data.shape[0])
batch = np.random.randint(final_test_data.shape[0], size=batch_size * 10)
batch_data = final_test_data[batch]

validation_set, val_context_words, val_cqas = prep_batch(batch_data, True)


# training_iterations_count: The number of data pieces to train on in total
# batch_size: The number of data pieces per batch
def train(iterations, batch_size):
    def Average(lst):
        return sum(lst) / len(lst)

    training_iterations = range(0, iterations, batch_size)
    acc10 = []
    tmp_loss10 = []
    wordz = []
    for j in tqdm(training_iterations):

        batch = np.random.randint(final_train_data.shape[0], size=batch_size)
        batch_data = final_train_data[batch]

        sess.run([opt_op], feed_dict=prep_batch(batch_data))
        if (j / batch_size) % display_step == 0:

            # Calculate batch accuracy
            acc, ccs, tmp_loss, log, con, cor, loc = sess.run([corrects, cs, total_loss, logit,
                                                               context_placeholder, corr, locs],
                                                              feed_dict=validation_set)
            if (j / batch_size) + max(100, int((j / batch_size) * 0.01)) >= len(training_iterations) and len(
                    training_iterations) > 100:
                # Calculate batch accuracy
                acc10.append(acc)
                # Calculate batch loss
                tmp_loss10.append(tmp_loss)
            # Display results
            print("Iter " + str(j / batch_size) + ", Minibatch Loss= ", tmp_loss,
                  "Accuracy= ", np.mean(acc))
    train_accuracy = np.mean(acc)
    return sess, train_accuracy


def restore_sess(location):
    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(session, location)
    return session


def session_save(location, iter, batch_size):
    saver = tf.train.Saver()
    session, train_accuracy = train(iter, batch_size)  # Small amount of training for preliminary results
    f = open("Train=" + str(train_accuracy), 'w')
    f.close()
    saver.save(session, location)
    return session


def session_manage(location, rewrite=False, iter=30000, batch_size=batch_size):
    full_location = location + "model.ckpt"
    if rewrite or not os.path.isdir(location):
        return session_save(full_location, iter, batch_size)
    else:
        return restore_sess(full_location)


train_location = "./neural/pre_trained_model/"
sess = session_manage(train_location, rewrite=True)
'''-----------------------------------------------------------'''
ancr = sess.run([corrbool, locs, total_loss, logits, facts_0s, w_1] + attends +
                [query, cs, question_module_outputs], feed_dict=validation_set)
a = ancr[0]
n = ancr[1]
cr = ancr[2]
attenders = np.array(ancr[6:-3])
faq = np.sum(ancr[4], axis=(-1, -2))  # Number of facts in each context

limit = 5
for question in range(min(limit, batch_size)):
    plt.yticks(range(passes, 0, -1))
    plt.ylabel("Episode")
    plt.xlabel("Question " + str(question + 1))
    pltdata = attenders[:, question, :int(faq[question]), 0]
    # Display only information about facts that actually exist, all others are 0
    pltdata = (pltdata - pltdata.mean()) / (pltdata.max() - pltdata.min() + 0.001) * 256
    plt.pcolor(pltdata, cmap=plt.cm.BuGn, alpha=0.7)
    plt.show()

# print(list(map((lambda x: x.shape),ancr[3:])), new_ends.shape)
'''-----------------------------------------------------------'''
# Locations of responses within contexts
indices = np.argmax(n, axis=1)

# Locations of actual answers within contexts
indicesc = np.argmax(a, axis=1)

for i, e, cw, cqa in list(zip(indices, indicesc, val_context_words, val_cqas))[:limit]:
    ccc = " ".join(cw)
    print("TEXT: ", ccc)
    print("QUESTION: ", " ".join(cqa[3]))
    print("RESPONSE: ", cw[i], ["Correct", "Incorrect"][i != e])
    print("EXPECTED: ", cw[e])
    print()
'''-----------------------------------------------------------'''
# train_location = "./max_train_model/"
# sess = session_manage(train_location, rewrite=True, iter=training_iterations_count, batch_size=batch_size)
'''-----------------------------------------------------------'''

# Final testing accuracy
print(np.mean(sess.run([corrects], feed_dict=prep_batch(final_test_data))[0]))
'''-----------------------------------------------------------'''
sess.close()
'''-----------------------------------------------------------'''
