import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib
import sys
import os
import zipfile
import tarfile
import json
import hashlib
import re
import itertools

train_set_file = "Data/SampleTestData"
test_set_file = "Data/SampleTrainData"
print("-------------------------------------------------------------------")
print("----------------------------Data path set--------------------------")
print("-------------------------------------------------------------------")
train_set_post_file = train_set_file
test_set_post_file = test_set_file

glove_vectors_file = "project/models/glove.6B.50d.txt"

# Deserialize GloVe vectors
glove_wordmap = {}
with open(glove_vectors_file, "r", encoding="utf8") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")
print("-------------------------------------------------------------------")
print("----------------------------Glove Model----------------------------")
print("-------------------------------------------------------------------")

wvecs = []
for item in glove_wordmap.items():
    wvecs.append(item[1])
s = np.vstack(wvecs)

v = np.var(s, 0)
m = np.mean(s, 0)
RS = np.random.RandomState()


def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk] = RS.multivariate_normal(m, np.diag(v))
    return glove_wordmap[unk]


def sentence2sequence(sentence):
    tokens = sentence.strip('"(),-').lower().split(" ")
    rows = []
    words = []
    # Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0:
            word = token[:i]
            #             print(word)
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


def contextualize(set_file):
    data = []
    context = []
    c = 0
    with open(set_file, "r", encoding="utf8") as train:
        for line in train:
            l, ine = tuple(line.split(" ", 1))

            #             print("l=", l)
            #             print("ine=", ine)
            # Split the line numbers from the sentences they refer to.
            if l is "1":
                # New contexts always start with 1,
                # so this is a signal to reset the context.
                context = []
            if "\t" in ine:
                # Tabs are the separator between questions and answers,
                # and are not present in context statements.
                question, answer, support = tuple(ine.split("\t"))

                data.append((tuple(zip(*context)) +
                             sentence2sequence(question) +
                             sentence2sequence(answer) +
                             ([int(s) for s in support.split()],)))
                # Multiple questions may refer to the same context, so we don't reset it.
            #                 print(len(data))
            else:
                # Context sentence.
                context.append(sentence2sequence(ine[:-1]))
    return data


print("-------------------------------------------------------------------")
print("----------------------------Word2Vec-------------------------------")
print("-------------------------------------------------------------------")

train_set_file = "project/models/SampleData"
test_set_file = "project/models/SampleData"

train_data = contextualize(train_set_post_file)
test_data = contextualize(train_set_post_file)

final_train_data = []


def finalize(data):
    final_data = []
    c = 0
    for cqas in data:
        if (len(cqas) != 7):
            print(cqas)
        contextvs, contextws, qvs, qws, avs, aws, spt = cqas
        if ((len(avs)) > 1):
            continue
        c = c + 1
        lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
        context_vec = np.concatenate(contextvs)
        context_words = sum(contextws, [])
        sentence_ends = np.array(list(lengths))
        final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws))
    return np.array(final_data)


print("-------------------------------------------------------------------")
print("-------------------Final Processing of Data------------------------")
print("-------------------------------------------------------------------")

final_train_data = finalize(train_data)
final_test_data = finalize(test_data)

tf.reset_default_graph()

# Hyperparameters

recurrent_cell_size = 128
D = 50
learning_rate = 0.005
input_p, output_p = 0.5, 0.5
batch_size = 128
passes = 4
ff_hidden_size = 256
weight_decay = 0.00000001
# ==========================================================================================================================
training_iterations_count = 40000
# training_iterations_count = 400000
display_step = 100

print("-------------------------------------------------------------------")
print("------------------------HyperParameters set------------------------")
print("-------------------------------------------------------------------")
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

query = tf.placeholder(tf.float32, [None, None, D], "query")

# input_query_lengths: A [batch_size, 2] tensor that contains question length information.
# input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range()
# so that it plays nice with gather_nd.
input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, query, dtype=tf.float32,
                                               scope=tf.VariableScope(True, "input_module"))

# q: the question states. A [batch_size, recurrent_cell_size] tensor.
q = tf.gather_nd(question_module_outputs, input_query_lengths)
# print(q.shape)


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
    with tf.variable_scope("attending") as scope:
        # attending: The metrics by which we decide what to attend to.
        attending = tf.concat([c, mem, re_q, c * re_q, c * mem, (c - re_q) ** 2, (c - mem) ** 2], 2)

        m1 = tf.matmul(attending * existing_facts,
                       tf.tile(w_1, tf.stack([tf.shape(attending)[0], 1, 1]))) * existing_facts
        bias_1 = b_1 * existing_facts
        tnhan = tf.nn.relu(m1 + bias_1)
        m2 = tf.matmul(tnhan, tf.tile(w_2, tf.stack([tf.shape(attending)[0], 1, 1])))
        bias_2 = b_2 * existing_facts
        norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)
        softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:, :-1]
        softmax_gather = tf.gather_nd(norm_m2[..., 0], softmax_idx)
        softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
        softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)), -1)


facts_0s = tf.cast(tf.count_nonzero(input_sentence_endings[:, :, -1:], -1, keepdims=True), tf.float32)

with tf.variable_scope("Episodes") as scope:
    attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)
    memory = [q]
    attends = []
    for a in range(passes):
        attend_to = attention(cs, tf.tile(tf.reshape(memory[-1], [-1, 1, recurrent_cell_size]), size),
                              facts_0s)
        retain = 1 - attend_to
        while_valid_index = (lambda state, index: index < tf.shape(cs)[1])
        update_state = (lambda state, index: (attend_to[:, index, :] *
                                              attention_gru(cs[:, index, :], state)[0] +
                                              retain[:, index, :] * state))
        memory.append(tuple(tf.while_loop(while_valid_index,
                                          (lambda state, index: (update_state(state, index), index + 1)),
                                          loop_vars=[memory[-1], 0]))[0])

        attends.append(attend_to)
        scope.reuse_variables()

# Answer Module

a0 = tf.concat([memory[-1], q], -1)

fc_init = tf.random_normal_initializer(stddev=0.1)
with tf.variable_scope("answer"):
    w_answer = tf.get_variable("weight", [recurrent_cell_size * 2, D],
                               tf.float32, initializer=fc_init)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                         tf.nn.l2_loss(w_answer))
    logit = tf.expand_dims(tf.matmul(a0, w_answer), 1)
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
        mask = tf.scan(tf.logical_xor, mask_ends, tf.ones_like(range_ind, dtype=bool))
    logits = -tf.reduce_sum(tf.square(context * tf.transpose(tf.expand_dims(
        tf.cast(mask, tf.float32), -1), [1, 0, 2]) - logit), axis=-1)

gold_standard = tf.placeholder(tf.float32, [None, 1, D], "answer")
with tf.variable_scope('accuracy'):
    eq = tf.equal(context, gold_standard)
    corrbool = tf.reduce_all(eq, -1)
    logloc = tf.reduce_max(logits, -1, keepdims=True)
    locs = tf.equal(logits, logloc)

    correctsbool = tf.reduce_any(tf.logical_and(locs, corrbool), -1)
    corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32),
                        tf.zeros_like(correctsbool, dtype=tf.float32))

    # corr: corrects, but for the right answer instead of our selected answer.
    corr = tf.where(corrbool, tf.ones_like(corrbool, dtype=tf.float32),
                    tf.zeros_like(corrbool, dtype=tf.float32))
with tf.variable_scope("loss"):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.nn.l2_normalize(logits, -1),
                                                   labels=corr)
    total_loss = tf.reduce_mean(loss) + weight_decay * tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
optimizer = tf.train.AdamOptimizer(learning_rate)
opt_op = optimizer.minimize(total_loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


def prep_batch(batch_data, more_data=False):
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
print("-------------------------------------------------------------------")
print("------------------------Training----------------------------------")
print("-------------------------------------------------------------------")

tqdm_installed = False
try:
    from tensorflow.python.ops.variables import Variable
    from tqdm import tqdm

    tqdm_installed = True
except:
    pass

batch = np.random.randint(final_test_data.shape[0], size=batch_size * 10)

batch_data = final_test_data[batch]

validation_set, val_context_words, val_cqas = prep_batch(batch_data, True)



def train(iterations, batch_size):
    training_iterations = range(0, iterations, batch_size)
    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    wordz = []
    for j in training_iterations:

        batch = np.random.randint(final_train_data.shape[0], size=batch_size)
        # batch_data = tf.placeholder(tf.float32, shape=[None, 9,None])
        batch_data = final_train_data[batch]
        feed_dict = prep_batch(batch_data)
        sess.run([opt_op], feed_dict=prep_batch(batch_data))

        if (j / batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc, ccs, tmp_loss, log, con, cor, loc = sess.run([corrects, cs, total_loss, logit,
                                                               context_placeholder, corr, locs],
                                                              feed_dict=validation_set)

# 3000 =================================================================
train(3000, batch_size)  # Small amount of training for preliminary results

ancr = sess.run([corrbool, locs, total_loss, logits, facts_0s, w_1] + attends +
                [query, cs, question_module_outputs], feed_dict=validation_set)
a = ancr[0]
n = ancr[1]
cr = ancr[2]
attenders = np.array(ancr[6:-3])
faq = np.sum(ancr[4], axis=(-1, -2))  # Number of facts in each context

indices = np.argmax(n, axis=1)

indicesc = np.argmax(a, axis=1)
limit = 5
text =[]
questionAsked = []
actual = []
predicted = []




for i, e, cw, cqa in list(zip(indices, indicesc, val_context_words, val_cqas))[:limit]:
    print("E ", e)
    ccc = " ".join(cw)
    print("TEXT: ", ccc)
    text.append(ccc)
    print("QUESTION: ", " ".join(cqa[3]))
    questionAsked.append(" ".join(cqa[3]))
    print("RESPONSE: ", cw[i], ["Correct", "Incorrect"][i != e])
    predicted.append(cw[i])
    print("EXPECTED: ", cw[e])
    actual.append(cw[e])
    print(i)


# f = open('inception.log', 'w')
# f.writelines(text)
# f.close()

train(training_iterations_count, batch_size)
