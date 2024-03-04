import json

remove_capitalization = True
remove_ending_questionmark = True

f = open('h_n_oos_conversation.json')
data = json.load(f)
f.close()

formatted_data = []
formatted_ins_data = []
total_count = 0
positive_count = 0
negative_count = 0
no_generation_count = 0
count_dict = {}
intent_list = []


for item in data:
    intent = item['intent']
    sentence = item['output']

    if intent not in intent_list:
        intent_list.append(intent)

    if remove_capitalization:
        sentence = sentence.lower()

    # Handle edge cases when ChatGPT still provides some padding for the response
    if 'Sure' in sentence or 'sure' in sentence:
        parts = sentence.split(':')
        if len(parts) == 2:
            sentence = parts[1]
        else:
            parts = sentence.split('! ')
            if len(parts) == 2:
                sentence = parts[1]
            else:
                parts = sentence.split('. ')
                if len(parts) == 2:
                    sentence = parts[1]
                else:
                    print("Special case encountered for: ", sentence)
                    sentence = input("Please manually enter the sentence: ")

    if item['in_scope'] == "n/a":
        pass
      
    # Remove the rare cases when ChatGPT is unable to generate a hard-negative OOS
    elif 'sorry' in sentence or "Sorry" in sentence or "apologize" in sentence:
        total_count += 1
        no_generation_count += 1
      
    # For the utterances that did not pass the step 1 of the 2-step OOS verification
    elif 'Yes' in item['in_scope']:
        total_count += 1
        positive_count += 1
        if "\"" in sentence:
            new_sentence = ""
            for token in sentence:
                if token != "\"":
                    new_sentence += token
            sentence = new_sentence
        if sentence[0] == " ":
            sentence = sentence[1:]
        if remove_ending_questionmark:
            if sentence[-1] == "?":
                sentence = sentence.rstrip(sentence[-1])
        formatted_ins_data.append([sentence, intent])

    # For the utterances that passed the step 1 of the the 2-step OOS verification
    elif "No" in item['in_scope']:
        total_count += 1
        negative_count += 1
        if "\"" in sentence:
            new_sentence = ""
            for token in sentence:
                if token != "\"":
                    new_sentence += token
            sentence = new_sentence
        if sentence[0] == " ":
            sentence = sentence[1:]
        if remove_ending_questionmark:
            if sentence[-1] == "?":
                sentence = sentence.rstrip(sentence[-1])
        formatted_data.append([sentence, intent])
        if intent not in count_dict:
            count_dict[intent] = 1
        else:
            count_dict[intent] += 1

    # Unexpected response
    else:
        print('the value is: ', item['in_scope'])

sorted_count_dict = sorted(count_dict.items(), key=lambda x:x[1])

statistics = {"intents": len(intent_list),
              "total count": total_count,
              "hard negative count": negative_count,
              "in scope count": positive_count,
              "no generation count": no_generation_count,
              "hard negative rate": negative_count/total_count,
              "hard negatives for each intent": dict(sorted_count_dict)
              }


result = {'hn': formatted_data, 'ins': formatted_ins_data, 'statistics': statistics}

# Write the results in a file
with open('organized_h_n.json', 'w') as f:
    json.dump(result, f, indent=4)

