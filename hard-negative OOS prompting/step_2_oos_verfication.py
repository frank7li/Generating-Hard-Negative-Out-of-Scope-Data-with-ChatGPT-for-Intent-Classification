import json
import openai
import time
import os


MODEL = 'gpt-3.5-turbo'

# Change to the file name that contains your key for the ChatGPT API
with open('keys.json', 'r') as f:
    _keys = json.load(f)

openai.api_key = _keys['openai']

# collect the generated hard negative OOS after step 1 of 2-step OOS verification
f = open('organized_h_n.json.json')
data = json.load(f)
f.close()


intent_list = []
for element in data["statistics"]["hard negatives for each intent"]:
    intent_list.append(element)


# Format the instruction used in the 'system' role
# Show ChatGPT the name of all the intents in the dataset
instruction = "Here are the list of intents: "
for i in range(0, len(intent_list)-1):
    instruction += intent_list[i]
    instruction += ", "

instruction += intent_list[-1]
instruction += ". I will give you a sentence and you will determine whether the sentence belong to one of the intents. If it does, reply the name of the intent it belongs with, if the sentence is out of scope of all the intents, reply 'no'."


result_file_name = 'step_2_gpt_scope_check.json'
if os.path.exists(result_file_name):
  f = open(result_file_name)
  finished_data = json.load(f)
  f.close()
else:
  finished_data = []


# create a list that contain hard-negative OOS that still needs to be checked
source = []
for i in range(len(data['hn'])):
    if finished_data == [] or i >= len(finished_data['hn']):
        source.append(data['hn'][i])

# Using the 'system' role to show ChatGPT the name of all the intents and guide its behavior
messages = [{'role': 'system', 'content': instruction}]

count = 0

# Checking with ChatGPT to see if each hard negative OOS is OOS with respect to the entire dataset
for sample in source:
    print('----------------------------------')
    count += 1
    print(count,"/", len(source))
    if len(sample) == 2:
        content = sample[0]
    else:
        print('error')
        break
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
    finished_data.append([
        content,
        sample[1],
        chat_response
    ])
    # save the results in a file
    with open(result_file_name, 'w') as f:
        json.dump(finished_data, f, indent=4)

    # this is added to avoid API for timing out due to too frequent requests
    print("waiting for 30 seconds")
    time.sleep(30)
    print()
