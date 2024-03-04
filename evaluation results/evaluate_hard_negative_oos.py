import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from ood_metrics import auroc, aupr, fpr_at_95_tpr

# Change the model name and dataset name to evaluate different datasets
model_name = "bert"
dataset_name = "clinc"

# load the hard-negative OOS data
file_name = "hard_negative_oos_" + dataset_name + ".json"
f = open(file_name)
data = json.load(f)
f.close()

hn_oos = []
intent_list = []

for item in data['h_n_oos']:
  hn_oos.append(item)
  if item[1] not in intent_list:
    intent_list.append(item[1])

# load the general OOS data
file_name = "oos_" + dataset_name + ".json"
f = open(file_name)
oos_data = json.load(f)
f.close()


oos_data_test = oos_data["oos"]

# load the INS data
file_name = "processed_" + dataset_name + ".json"
f = open(file_name)
data = json.load(f)
f.close()


X = []
y = []
for pair in data:
    if len(pair) == 2:
        if pair[1] in intent_list:
            X.append(pair[0])
            y.append(pair[1])
          
# Divide the data into 80/20 train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)



label_dict = {label: i for i, label in enumerate(set(y_train))}
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

# Training the model
model.fit(x_to_tensor, y_to_np_array, epochs=5, batch_size=32)

t = 1.0

def energy_function(logits):
    energy = -t * tf.math.log(tf.reduce_sum(tf.exp(logits / t), axis=1))
    return energy


# Testing for in-scope samples
in_scope_text_encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors='tf')
in_scope_text_to_tensor = in_scope_text_encoded['input_ids']
encoded_in_scope_labels = [label_dict[label] for label in y_test]
encoded_in_scope_labels = np.array(encoded_in_scope_labels)
predictions_in_scope = model.predict([in_scope_text_to_tensor])
logits = predictions_in_scope['logits']
softmax_in_scope = tf.nn.softmax(logits)
energy_conf_scores_in_scope = energy_function(logits)
confidence_score_in_scope = []
ins_confidence_result = []
for i in range(len(softmax_in_scope)):
    confidence = np.max(softmax_in_scope[i])
    energy_confidence = energy_conf_scores_in_scope[i].numpy()
    confidence_score_in_scope.append(confidence)
    text = X_test[i]
    intented_label = y_test[i]
    predicted_label_encoded = np.argmax(softmax_in_scope[i])
    predicted_label = number_to_label[predicted_label_encoded]
    ins_confidence_result.append({'utterance': text, 'true label': intented_label,'predicted label': predicted_label,'model confidence': str(confidence), 'energy confidence': str(energy_confidence)})





# Testing for hard-negative out-of-scope samples
hard_negative_text = [sample[0] for sample in hn_oos]
hard_negative_label = [sample[1] for sample in hn_oos]
hard_negative_text_encoded = tokenizer(hard_negative_text, padding=True, truncation=True, return_tensors='tf')
hard_negative_text_to_tensor = hard_negative_text_encoded['input_ids']
predictions_hard_negative = model.predict([hard_negative_text_to_tensor])
logits = predictions_hard_negative['logits']
softmax_hard_negative = tf.nn.softmax(logits)
energy_conf_scores_hard_negative = energy_function(logits)
confidence_score_hard_negative = []
confidence_per_sentence = []
for i in range(len(softmax_hard_negative)):
    confidence = np.max(softmax_hard_negative[i])
    energy_confidence = energy_conf_scores_hard_negative[i].numpy()
    confidence_score_hard_negative.append(confidence)
    text = hard_negative_text[i]
    intented_label = hard_negative_label[i]
    predicted_label_encoded = np.argmax(softmax_hard_negative[i])
    predicted_label = number_to_label[predicted_label_encoded]
    confidence_per_sentence.append({'utterance': text, 'intented label': intented_label,'predicted label': predicted_label,'model confidence': str(confidence), 'energy confidence': str(energy_confidence)})




# Testing for general OOS samples
oos_text = [sample[0] for sample in oos_data_test]
oos_label = [sample[1] for sample in oos_data_test]
oos_text_encoded = tokenizer(oos_text, padding=True, truncation=True, return_tensors='tf')
oos_text_to_tensor = oos_text_encoded['input_ids']
predictions_oos = model.predict([oos_text_to_tensor])
logits = predictions_oos['logits']
softmax_oos = tf.nn.softmax(logits)
energy_conf_scores_oos = energy_function(logits)
confidence_score_oos = []
oos_confidence_result = []
for i in range(len(softmax_oos)):
    confidence = np.max(softmax_oos[i])
    energy_confidence = energy_conf_scores_oos[i].numpy()
    confidence_score_oos.append(confidence)
    text = oos_text[i]
    predicted_label_encoded = np.argmax(softmax_oos[i])
    predicted_label = number_to_label[predicted_label_encoded]
    oos_confidence_result.append({'utterance': text, 'predicted label': predicted_label,'model confidence': str(confidence), 'energy confidence': str(energy_confidence)})


metrics_scores = {}

# Evaluating AUROC, AUPR, FP95
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


# Evaluating accuracy, precision, recall, and F1 at selected confidence thresholds

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


# Evaluating energy scores with AUROC, AUPR, and FP95
energy_metrics_scores = {}

# traditional approach
binary_labels = []
binary_scores = []
for element in energy_conf_scores_in_scope:
    binary_labels.append(1)
    binary_scores.append(element)

for element in energy_conf_scores_hard_negative:
    binary_labels.append(0)
    binary_scores.append(element)

auroc_score = auroc(binary_scores, binary_labels)
aupr_score = aupr(binary_scores, binary_labels)
fp95_score = fpr_at_95_tpr(binary_scores, binary_labels)
ins_vs_hn = {'AUROC': str(auroc_score), "AUPR": str(aupr_score), "fp95": str(fp95_score)}


binary_labels = []
binary_scores = []
for element in energy_conf_scores_oos:
    binary_labels.append(0)
    binary_scores.append(element)

for element in energy_conf_scores_hard_negative:
    binary_labels.append(1)
    binary_scores.append(element)

auroc_score = auroc(binary_scores, binary_labels)
aupr_score = aupr(binary_scores, binary_labels)
fp95_score = fpr_at_95_tpr(binary_scores, binary_labels)

oos_vs_hn = {'AUROC': str(auroc_score), "AUPR": str(aupr_score), "fp95": str(fp95_score)}


binary_labels = []
binary_scores = []
for element in energy_conf_scores_in_scope:
    binary_labels.append(1)
    binary_scores.append(element)

for element in energy_conf_scores_oos:
    binary_labels.append(0)
    binary_scores.append(element)

auroc_score = auroc(binary_scores, binary_labels)
aupr_score = aupr(binary_scores, binary_labels)
fp95_score = fpr_at_95_tpr(binary_scores, binary_labels)

ins_vs_oos = {'AUROC': str(auroc_score), "AUPR": str(aupr_score), "fp95": str(fp95_score)}

energy_metrics_scores['traditional'] = {'ins_vs_hn': ins_vs_hn, 'oos_vs_hn': oos_vs_hn, 'ins_vs_oos':ins_vs_oos}

# Calculate average confidence scores
softmax = {'INS':str(np.mean(confidence_score_in_scope)), 'HN_OOS':str(np.mean(confidence_score_hard_negative)), 'OOS':str(np.mean(confidence_score_oos))}
energy = {'INS':str(np.mean(energy_conf_scores_in_scope)), 'HN_OOS':str(np.mean(energy_conf_scores_hard_negative)), 'OOS':str(np.mean(energy_conf_scores_oos))}
average_conf={'softmax': softmax, 'energy': energy}

# Save all the results
result = {'softmax metric scores': metrics_scores, 'energy scores': energy_metrics_scores['traditional'], 'average confidence scores': average_conf, 'confidence HN OOS': confidence_per_sentence, 'confidence INS': ins_confidence_result, 'confidence OOS': oos_confidence_result}
file_name = "results_"+ model_name + "_" + dataset_name + ".json"
with open(file_name, 'w') as f:
    json.dump(result, f, indent=4)


