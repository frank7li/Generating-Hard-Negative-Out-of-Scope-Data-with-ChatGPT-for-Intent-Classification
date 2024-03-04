import json
import itertools
import openai
import time
import os


# Note, this program executes both the hard-negative OOS generation step and step 1 of the 2-step OOS verfications step

# Edit this variable to the name of the file that contains the INS training data
# The in-scope training data stored in a json file that contain a list of tuples that contain [utterance, intent]
dataset_name = 'in_scope_data.json'

MODEL = 'gpt-3.5-turbo'

# Edit this file name to where your ChatGPT API key is stored
with open('keys.json', 'r') as f:
    _keys = json.load(f)

openai.api_key = _keys['openai']


# Where the dialogue between you and ChatGPT are stored
result_file_name = "h_n_oos_conversation.json"

# Check if there is already hard-negative OOS generated
if os.path.exists(result_file_name):
  first_generation = False
  f = open(result_file_name)
    data = json.load(f)
    f.close()
else:
  first_generation = True
  data = []

    

# Collect the intents that have hard-negative OOS generated
finished_intents = []
for element in data:
    intent = element['intent']
    if intent not in finished_intents:
        finished_intents.append(intent)

# collect keywords
f = open("frequent_words.json")
keywords = json.load(f)
f.close()


# collect training data
def collecting_training_data(file_name):
    f = open(file_name)
    dataset = json.load(f)
    f.close()
    return dataset

dataset = collect_training_data(dataset_name)

# Checking to see if this program was terminated last time before all 40 hard-negative OOS per intent has been generated
last_intent = finished_intents[-1]
last_finished_count = 0
for element in data:
    if element['intent'] == last_intent:
        last_finished_count += 1

skip = False if last_finished_count == (40 or 0) else True


# Create a dict that contains relevant data used to form prompts
source = {}
for intent in keywords:
    if (intent not in finished_intents) or (intent == last_intent and skip):
        words = keywords[intent]
        source[intent] = {}
        source[intent]['words'] = words


# Adding 5 examples INS utterances per intent into source, are displayed to ChatGPT later
for element in dataset:
    if len(element) == 2:
        intent = element[1]
        if (intent not in finished_intents) or (intent == last_intent and skip):
            sentence = element[0]
            if 'samples' not in source[intent].keys():
                source[intent]['samples'] = [sentence]
            elif len(source[intent]['samples']) < 5:
                source[intent]['samples'].append(sentence)

chat_response = ""

count = len(finished_intents) if not skip else len(finished_intents)-1
total_count = len(source) + count if not skip else len(source) + count - 1

for intent in source:
    if "+" in intent or " " in intent:
        break
    print('---------------------------------------------------------------')
    count += 1
    print(count, "/", total_count)
    print('----------------------------------')

    # using the system role to guide ChatGPT's behavior
    messages = [{'role': 'system', 'content': 'i will first show you the name of an intent and 5 in scope samples, just try to understand this intent. Only reply ok unless I am asking you to generate a question. Keep the response short'}]
    words = source[intent]['words']
    samples = source[intent]['samples']
    if len(samples) < 5:
        break

    # showing ChatGPT 5 INS samples to understand the intent
    content = "Here are 5 samples from the intent '{}': '{}', '{}', '{}', '{}', '{}'."
    content = content.format(intent, samples[0], samples[1], samples[2], samples[3], samples[4])
    if skip:
        count_sample = last_finished_count+1
    else:
        count_sample = 1
    messages.append({'role': 'user', 'content': content})
    word_combinations = list(itertools.combinations(words, 2))
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages
    )
    chat_response = completion.choices[0].message.content
    print(f'User: {content}')
    print()
    print(f'ChatGPT: {chat_response}')
    print()
    messages.append({'role': 'assistant', 'content': chat_response})
    count = 0

    # begin prompting for hard-negative OOS
    for group in word_combinations:
        i = 0
        while (i < 4):
            if skip and count < last_finished_count:
                count += 1
                i += 1
            else:
                if i == 0:
                    content = "give me a question with fewer than 15 tokens that has to include both of the words '{}' and '{}', and the question is not related to this '{}' intent."
                else:
                    content = "give me another question with fewer than 15 tokens that has to include both of the words '{}' and '{}', and the question is not related to this '{}' intent."
                content = content.format(group[0], group[1], intent)
                messages.append({'role': 'user', 'content': content})
                completion = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=messages
                )
                chat_response = completion.choices[0].message.content
                print("prompt #", count_sample)
                count_sample += 1
                print(f'User: {content}')
                print()
                print(f'ChatGPT: {chat_response}')
                print()
                messages.append({'role': 'assistant', 'content': chat_response})
                sentence = chat_response
                prompt = content

                # perform step 1 for the 2-step OOS verification
                content = "answer in 'yes' or 'no', does the last sentence you just generated related to this '{}' intent?"
                content = content.format(intent)
                messages.append({'role': 'user', 'content': content})
                completion = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=messages
                )
                chat_response = completion.choices[0].message.content
                print(f'User: {content}')
                print()
                print(f'ChatGPT: {chat_response}')
                print()
                messages.append({'role': 'assistant', 'content': chat_response})
                data.append({
                    'prompt': prompt,
                    'intent': intent,
                    'output': sentence,
                    'in_scope': chat_response,
                })
                with open(result_file_name, 'w') as f:
                    json.dump(data, f, indent=4)
                i += 1
                skip = False
                print("progress saved")
                print("-----------------------")

                # this is added to avoid API for timing out due to too frequent requests
                print("waiting for 30 seconds")
                time.sleep(30)
            

