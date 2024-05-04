# -*- coding: utf-8 -*-
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from scipy import linalg
from collections import defaultdict

def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def get_dict_0(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        en = my_file.loc[i][0]
        fr = my_file.loc[i][1]
        etof[en] = fr

    return etof


def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10    
    dot = np.dot(A, B)
    normb = np.linalg.norm(B)
    
    if len(A.shape) == 1: # If A is just a vector, we get the norm
        norma = np.linalg.norm(A)
        cos = dot / (norma * normb)
    else: # If A is a matrix, then compute the norms of the word vectors of the matrix (norm of each row)
        norma = np.linalg.norm(A, axis=1)
        epsilon = 1.0e-9 # to avoid division by 0
        cos = dot / (norma * normb + epsilon)
        
    return cos


def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx


def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed


def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data) - C:
            print("i is being set to", C)
            i = C


def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        if len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []




def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    ### START CODE HERE ###
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T


def get_dict(data):
    """
    Input:
        K: the number of negative samples
        data: the data you want to pull from
        indices: a list of word indices
    Output:
        word_dict: a dictionary with the weighted probabilities of each word
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    #
    #     words = nltk.word_tokenize(data)
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word

#############################################################################

import string
import re
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples, wordnet 
from nltk.stem import WordNetLemmatizer

stopwords_english = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def process_tweet_2(tweet):
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    
    '''
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = nltk.pos_tag(tokenizer.tokenize(tweet))

    tweets_clean = []
    for word in tweet_tokens:
        if (word[0] not in stopwords_english and # remove stopwords
            word[0] not in string.punctuation): # remove punctuation
            stem_word = lemmatizer.lemmatize(word[0], pos_tag_convert(word[1]))
            tweets_clean.append(stem_word)
    return tweets_clean


def pos_tag_convert(nltk_tag: str) -> str:
    '''Converts nltk tags to tags that are understandable by the lemmatizer.
    
    Args:
        nltk_tag (str): nltk tag
        
    Returns:
        _ (str): converted tag
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.NOUN

    
def load_tweets():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')  
    return all_positive_tweets, all_negative_tweets


############################################################################

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import pathlib

path_to_file = pathlib.Path("/content/NLP_Chapitre/data/NLP with Attention models  Translation Machine/por.txt")

np.random.seed(1234)
tf.random.set_seed(1234)

def load_data(path):
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    context = np.array([context for target, context, _ in pairs])
    target = np.array([target for target, context, _ in pairs])

    return context, target


portuguese_sentences, english_sentences = load_data(path_to_file)

sentences = (portuguese_sentences, english_sentences)

BUFFER_SIZE = len(english_sentences)
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(portuguese_sentences),)) < 0.8

train_raw = (
    tf.data.Dataset.from_tensor_slices(
        (english_sentences[is_train], portuguese_sentences[is_train])
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
val_raw = (
    tf.data.Dataset.from_tensor_slices(
        (english_sentences[~is_train], portuguese_sentences[~is_train])
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)


def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


max_vocab_size = 12000

english_vectorizer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True
)

english_vectorizer.adapt(train_raw.map(lambda context, target: context))

portuguese_vectorizer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True
)

portuguese_vectorizer.adapt(train_raw.map(lambda context, target: target))


def process_text(context, target):
    context = english_vectorizer(context).to_tensor()
    target = portuguese_vectorizer(target)
    targ_in = target[:, :-1].to_tensor()
    targ_out = target[:, 1:].to_tensor()
    return (context, targ_in), targ_out


train_data = train_raw.map(process_text, tf.data.AUTOTUNE)
val_data = val_raw.map(process_text, tf.data.AUTOTUNE)

del train_raw
del val_raw


def masked_loss(y_true, y_pred):
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    # Check which elements of y_true are padding
    mask = tf.cast(y_true != 0, loss.dtype)
    
    loss *= mask
    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result


#########################################################################################
import pandas as pd
import re


def get_train_test_data(data_dir):
    # Get the train data
    train_data = pd.read_json(f"{data_dir}/train.json")
    train_data.drop(['id'], axis=1, inplace=True)

    # Get the test data
    test_data = pd.read_json(f"{data_dir}/test.json")
    test_data.drop(['id'], axis=1, inplace=True)
    
    return train_data, test_data


def preprocess(input_data):
    # Define the custom preprocessing function
    def preprocess_util(input_data):
        # Convert all text to lowercase
        lowercase = input_data.lower()
        # Remove newlines and double spaces
        removed_newlines = re.sub("\n|\r|\t", " ",  lowercase)
        removed_double_spaces = ' '.join(removed_newlines.split(' '))
        # Add start of sentence and end of sentence tokens
        s = '[SOS] ' + removed_double_spaces + ' [EOS]'
        return s
    
    # Apply the preprocessing to the train and test datasets
    input_data['summary'] = input_data.apply(lambda row : preprocess_util(row['summary']), axis = 1)
    input_data['dialogue'] = input_data.apply(lambda row : preprocess_util(row['dialogue']), axis = 1)

    document = input_data['dialogue']
    summary = input_data['summary']
    
    return document, summary









