import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from ood_metrics import auroc, aupr, fpr_at_95_tpr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "bert"
dataset_name = "banking77"

# collect hard-negative OOS data
file_name = "hard_negative_oos_" + dataset_name + ".json"
f = open(file_name)
data = json.load(f)
f.close()

hn_oos = data['h_n_oos']


# collect general OOS data
file_name = "oos_" + dataset_name + ".json"
f = open(file_name)
oos = json.load(f)
f.close()

oos_data = oos["oos"]


hn_sentence = [i[0] for i in hn_oos]
hn_label = ['oos' for i in hn_oos]
hn_sentence_train, hn_sentence_test, hn_label_train, hn_label_test = train_test_split(hn_sentence, hn_label, train_size=0.8)

# load in-scope data
file_name = dataset_name + ".json"
f = open(file_name)
data = json.load(f)
f.close()


X = []
y = []
for pair in data:
    X.append(pair[0])
    y.append(pair[1])


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

X_train = X_train + hn_sentence_train
y_train = y_train + hn_label_train

y = y_train + y_test

label_dict = {label: i for i, label in enumerate(set(y))}
num_labels = len(label_dict)

if model_name == "roberta":
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
elif model_name == 'bert':
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
elif model_name == 'distilbert':
  tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
  model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
else:
  raise ("unsupported model name")


encoded_texts = tokenizer(X_train, padding=True, truncation=True, return_tensors='tf')

# converting tokenized batch encoding to tensors
x_to_tensor = encoded_texts['input_ids']

# tokenizing the labels
number_to_label = {}
for label in label_dict:
    number_to_label[label_dict[label]] = label
encoded_labels = [label_dict[label] for label in y_train]

# converting the labels to a numpy array
y_to_np_array = np.array(encoded_labels)

optimizer = tf.keras.optimizers.Adam(learning_rate=4e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.fit(x_to_tensor, y_to_np_array, epochs=5, batch_size=32)


# find the index for the encoding for oos
oos_index = label_dict['oos']

# testing for in scope samples
in_scope_text_encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors='tf')
in_scope_text_to_tensor = in_scope_text_encoded['input_ids']
encoded_in_scope_labels = [label_dict[label] for label in y_test]
encoded_in_scope_labels = np.array(encoded_in_scope_labels)
score_in_scope = model.evaluate(in_scope_text_to_tensor, encoded_in_scope_labels)
loss_in_scope = score_in_scope[0]
accuracy_in_scope = score_in_scope[1]
predictions_in_scope = model.predict([in_scope_text_to_tensor])
logits = predictions_in_scope['logits']
softmax_in_scope = tf.nn.softmax(logits)
confidence_score_in_scope = []
ins_confidence_result = []
confidence_oos_for_ins = []
for i in range(len(softmax_in_scope)):
    softmax_arr = softmax_in_scope[i].numpy()
    # confidence2 is how confident the model predicts each utterance to be OOS
    confidence2 = softmax_arr[oos_index]
    confidence_oos_for_ins.append(confidence2)
    new_softmax = np.delete(softmax_arr, oos_index)
    confidence = np.max(new_softmax)
    confidence_score_in_scope.append(confidence)
    text = X_test[i]
    intented_label = y_test[i]
    predicted_label_encoded = np.argmax(softmax_in_scope[i])
    predicted_label = number_to_label[predicted_label_encoded]
    ins_confidence_result.append({'utterance': text, 'true label': intented_label,'predicted label': predicted_label,'confidence ins': str(confidence), 'confidence oos': str(confidence2)})





# testing for hard-negative OOS
hard_negative_text_encoded = tokenizer(hn_sentence_test, padding=True, truncation=True, return_tensors='tf')
hard_negative_text_to_tensor = hard_negative_text_encoded['input_ids']
encoded_hard_negative_labels = [label_dict[label] for label in hn_label_test]
encoded_hard_negative_labels = np.array(encoded_hard_negative_labels)
score_hard_negative = model.evaluate(hard_negative_text_to_tensor, encoded_hard_negative_labels)
loss_hard_negative = score_hard_negative[0]
accuracy_hard_negative = score_hard_negative[1]
predictions_hard_negative = model.predict([hard_negative_text_to_tensor])
logits = predictions_hard_negative['logits']
softmax_hard_negative = tf.nn.softmax(logits)
confidence_score_hard_negative = []
confidence_per_sentence = []
confidence_to_be_hard_negative = []
for i in range(len(softmax_hard_negative)):
    softmax_arr = softmax_hard_negative[i].numpy()
    confidence2 = softmax_arr[oos_index]
    confidence_to_be_hard_negative.append(confidence2)
    new_softmax = np.delete(softmax_arr, oos_index)
    confidence = np.max(new_softmax)
    confidence_score_hard_negative.append(confidence)
    text = hn_sentence_test[i]
    true_label = 'oos'
    predicted_label_encoded = np.argmax(softmax_hard_negative[i])
    predicted_label = number_to_label[predicted_label_encoded]
    confidence_per_sentence.append({'utterance': text, 'true label': true_label,'predicted label': predicted_label,'confidence ins': str(confidence), 'confidence oos': str(confidence2)})



# testing for general oos
oos_text = [sample[0] for sample in oos_data]
oos_label = [sample[1] for sample in oos_data]
oos_text_encoded = tokenizer(oos_text, padding=True, truncation=True, return_tensors='tf')
oos_text_to_tensor = oos_text_encoded['input_ids']
encoded_oos_labels = [label_dict[label] for label in oos_label]
encoded_oos_labels = np.array(encoded_oos_labels)
score_oos = model.evaluate(oos_text_to_tensor, encoded_oos_labels)
loss_oos = score_oos[0]
accuracy_oos = score_oos[1]
predictions_oos = model.predict([oos_text_to_tensor])
logits = predictions_oos['logits']
softmax_oos = tf.nn.softmax(logits)
confidence_score_oos = []
oos_confidence_result = []
confidence_to_be_oos = []
for i in range(len(softmax_oos)):
    softmax_arr = softmax_oos[i].numpy()
    confidence2 = softmax_arr[oos_index]
    confidence_to_be_oos.append(confidence2)
    new_softmax = np.delete(softmax_arr, oos_index)
    confidence = np.max(new_softmax)
    confidence_score_oos.append(confidence)
    text = oos_text[i]
    true_label = 'oos'
    predicted_label_encoded = np.argmax(softmax_oos[i])
    predicted_label = number_to_label[predicted_label_encoded]
    oos_confidence_result.append({'utterance': text, 'true label': true_label,'predicted label': predicted_label,'confidence ins': str(confidence), 'confidence oos': str(confidence2)})

print("Accuracy in scope from " + dataset_name + ": ", accuracy_in_scope)
print("Accuracy hard negative OOS from " + dataset_name + ": ", accuracy_hard_negative)
print("Accuracy general oos from " + dataset_name + ": ", accuracy_oos)


metrics_scores = {}

# calculating AUROC, AUPR, and FP95
binary_labels = []
binary_scores = []
for element in confidence_score_in_scope:
    binary_labels.append(1)
    binary_scores.append(element)

for element in confidence_score_hard_negative:
    binary_labels.append(0)
    binary_scores.append(element)

auroc_score = auroc(binary_scores, binary_labels)
aupr_score = aupr(binary_scores, binary_labels)
fp95_score = fpr_at_95_tpr(binary_scores, binary_labels)
ins_vs_hn = {'AUROC': str(auroc_score), "AUPR": str(aupr_score), "fp95": str(fp95_score)}


binary_labels = []
binary_scores = []
for element in confidence_score_oos:
    binary_labels.append(0)
    binary_scores.append(element)

for element in confidence_score_hard_negative:
    binary_labels.append(1)
    binary_scores.append(element)

auroc_score = auroc(binary_scores, binary_labels)
aupr_score = aupr(binary_scores, binary_labels)
fp95_score = fpr_at_95_tpr(binary_scores, binary_labels)

oos_vs_hn = {'AUROC': str(auroc_score), "AUPR": str(aupr_score), "fp95": str(fp95_score)}


binary_labels = []
binary_scores = []
for element in confidence_score_in_scope:
    binary_labels.append(1)
    binary_scores.append(element)

for element in confidence_score_oos:
    binary_labels.append(0)
    binary_scores.append(element)

auroc_score = auroc(binary_scores, binary_labels)
aupr_score = aupr(binary_scores, binary_labels)
fp95_score = fpr_at_95_tpr(binary_scores, binary_labels)

ins_vs_oos = {'AUROC': str(auroc_score), "AUPR": str(aupr_score), "fp95": str(fp95_score)}

metrics_scores['traditional'] = {'ins_vs_hn': ins_vs_hn, 'oos_vs_hn': oos_vs_hn, 'ins_vs_oos':ins_vs_oos}


# calculate accuracy, precision, recall, f1 with confidence thresholds

metrics_scores['thresholding'] = {}
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for threshold in thresholds:

    binary_labels = []
    binary_scores = []
    for element in confidence_score_in_scope:
      binary_labels.append(1)
      if element >= threshold:
        binary_scores.append(1)
      else:
        binary_scores.append(0)

    for element in confidence_score_hard_negative:
      binary_labels.append(0)
      if element >= threshold:
        binary_scores.append(1)
      else:
        binary_scores.append(0)


    accuracy = accuracy_score(binary_labels, binary_scores)
    precision = precision_score(binary_labels, binary_scores)
    recall = recall_score(binary_labels, binary_scores)
    f1 = f1_score(binary_labels, binary_scores)
    ins_vs_hn = {'accuracy': str(accuracy), "precision": str(precision), "recall": str(recall), "f1": str(f1)}

    binary_labels = []
    binary_scores = []
    for element in confidence_score_in_scope:
      binary_labels.append(1)
      if element >= threshold:
        binary_scores.append(1)
      else:
        binary_scores.append(0)

    for element in confidence_score_oos:
      binary_labels.append(0)
      if element >= threshold:
        binary_scores.append(1)
      else:
        binary_scores.append(0)

    accuracy = accuracy_score(binary_labels, binary_scores)
    precision = precision_score(binary_labels, binary_scores)
    recall = recall_score(binary_labels, binary_scores)
    f1 = f1_score(binary_labels, binary_scores)
    ins_vs_oos = {'accuracy': str(accuracy), "precision": str(precision), "recall": str(recall), "f1": str(f1)}
    metrics_scores['thresholding'][threshold] = {'ins_vs_hn': ins_vs_hn, 'ins_vs_oos':ins_vs_oos}


average_conf = {'INS':str(np.mean(confidence_score_in_scope)), 'HN_OOS':str(np.mean(confidence_score_hard_negative)), 'OOS':str(np.mean(confidence_score_oos))}
result = {'average confidence scores': average_conf, 'scores': metrics_scores, 'confidence HN OOS': confidence_per_sentence, 'confidence INS': ins_confidence_result, 'confidence OOS': oos_confidence_result}

file_name = "results_"+ model_name + "_" + dataset_name + "_hn_training.json"
with open(file_name, 'w') as f:
    json.dump(result, f, indent=4)
