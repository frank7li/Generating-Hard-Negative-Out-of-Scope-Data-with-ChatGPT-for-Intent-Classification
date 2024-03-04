import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('wordnet')
nltk.download('stopwords')


# tokenize the utterances
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


# remove stopwords
def removing_stop_words(tokens):
    new_tokens = []
    for token in tokens:
        if token not in stopWords:
            if len(token) > 1:
                new_tokens.append(token)
    return new_tokens


# calculate token frequency
def token_frequency(data):
    intent_to_dict = {}
    for sample in data:
        if len(sample) == 2:
            intent = sample[1]
            if intent not in intent_to_dict:
                intent_to_dict[intent] = {}
            tokens = tokenize(sample[0])
            tokens = lemmatize_all_tokens(tokens)
            tokens = removing_stop_words(tokens)
            for token in tokens:
                if len(token)>2:
                    if token in intent_to_dict[intent]:
                        count = intent_to_dict[intent][token]
                        intent_to_dict[intent].update({token: count + 1})
                    else:
                        intent_to_dict[intent][token] = 1

    return intent_to_dict


# sort token frequency
def sort_by_value(intent_to_dict):
    sorted_dict = {}
    for intent in intent_to_dict:
        freq_dict = intent_to_dict[intent]
        sorted_freq_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        freq_dict = dict(sorted_freq_list)
        sorted_dict[intent] = freq_dict
    return sorted_dict


# calculate most frequent words
def most_frequent_words():
    data = collecting_training_data()
    dictionary = token_frequency(data)
    dictionary = sort_by_value(dictionary)
    return dictionary


# store most frequent words in a file
def write_most_frequent(word_dict, N, file_name):
    file = {}
    for intent in word_dict:
        freq_dict = word_dict[intent]
        sliced_dict = dict(list(freq_dict.items())[0: N])
        word_list = []
        for word in sliced_dict:
            word_list.append(word)
        file[intent] = word_list
    with open(file_name, 'w') as f:
        json.dump(file, f, indent=4)


# collect training data
def collecting_training_data(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data


# create a dict that contains a sorted dict of frequent words
# file_name should contain the in-scope training data stored in a json file that contain a list of tuples that contain [utterance, intent]
dict = most_frequent_words(filename)

# write 5 most frequent words in a file
write_most_frequent(dict, 5, 'frequent_words.json')
