import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eli5


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


stopWords = set(stopwords.words('english'))
vectorizer = CountVectorizer()
wordnet_lemmatizer = WordNetLemmatizer()


# tokenize the utterance
def tokenize(text):
    text = text.lower()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")

    tokens = text.split()
    return tokens

# lemmatize tokens
def lemmatize_all_tokens(tokens):
    new_tokens = []
    for token in tokens:
        new_tokens.append(wordnet_lemmatizer.lemmatize(token))
    return new_tokens


# removing stop words
def removing_stop_words(tokens):
    new_tokens = []
    for token in tokens:
        if token not in stopWords:
            if len(token) > 1:
                new_tokens.append(token)
    return new_tokens


# collect training data
def collecting_training_data(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data


# encode the sentences with one hot encoding, training the linearSVC, using Eli5 to find the keywords
def analyze_weights(file_name):
    data = collecting_training_data(file_name)
    count = 0
    intent_count = 0
    # forward mapping to allow encoding words and intent into one hot encoding
    word_to_number = {}
    intent_to_number = {}
    # backwards mapping for decoding
    number_to_word = {}
    number_to_intent = {}
    encoded_samples = []
    encoded_labels = []

    random.shuffle(data)

    # first traversal to record all the unique tokens
    for sample in data:
        intent = sample[1]
        if intent not in intent_to_number:
            intent_to_number[intent] = intent_count
            number_to_intent[intent_count] = intent
            intent_count += 1
        tokens = tokenize(sample[0])
        tokens = lemmatize_all_tokens(tokens)
        tokens = removing_stop_words(tokens)
        for token in tokens:
            if token not in word_to_number:
                word_to_number[token] = count
                number_to_word[count] = token
                count += 1

    size = len(word_to_number)

    # encode the labels and the samples in one hot encoding
    for sample in data:
        intent = sample[1]
        encoded_labels.append(intent_to_number[intent])
        one_hot_encoding = [0] * size
        tokens = tokenize(sample[0])
        tokens = lemmatize_all_tokens(tokens)
        tokens = removing_stop_words(tokens)
        for token in tokens:
            position = word_to_number[token]
            one_hot_encoding[position] = 1
        encoded_samples.append(one_hot_encoding)

    encoded_samples = np.array(encoded_samples)
    encoded_labels = np.array(encoded_labels)

    # training LinearSVC
    clf = make_pipeline(StandardScaler(), LinearSVC())
    clf.fit(encoded_samples, encoded_labels)
    result = eli5.formatters.text.format_as_text(eli5.explain_weights(clf))
    with open('eli5_result.txt', 'w') as f:
        f.write(result)

    keywords = process_eli5_result('eli5_result.txt')
    write_eli5_results(number_to_word, number_to_intent, keywords, 'eli5_result_keywords.txt')


# collect the results from eli5
def process_eli5_result(file_name):
    f = open(file_name)
    line = f.readline()
    result = []
    while line:
        line = line.rstrip()
        if line == '------  -------':
            subresult = []
            for i in range(4):
                line = f.readline()
                line = line.rstrip()
                info = line.split('x')
                if len(info) == 2:
                    word = info[1]
                    subresult.append(word)
            result.append(subresult)
        line = f.readline()
    f.close()
    return result


# write the highest weighted words from eli5 to 'eli5_result_keywords.txt;
def write_eli5_results(number_to_word, number_to_intent, word_list, file_name):
    file = {}
    count = 0
    for group in word_list:
        intent = number_to_intent[count]
        new_word_list = []
        for number in group:
            new_word_list.append(number_to_word[int(number)])
        file[intent] = new_word_list
        count += 1

    with open(file_name, 'w') as f:
        json.dump(file, f, indent=4)


# note, as explained in the paper, this step is not used in generating hard negative OOS utterances
# create a file that contains a sorted dict of frequent words
# file_name should contain the in-scope training data stored in a json file that contain a list of tuples that contain [utterance, intent]
analyze_weights(file_name)


