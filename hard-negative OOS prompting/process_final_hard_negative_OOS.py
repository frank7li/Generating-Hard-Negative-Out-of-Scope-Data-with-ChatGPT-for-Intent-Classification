import json

# Collect the file containing dialogue with ChatGPT for step 2 in 2-step OOS verification
f = open('step_2_gpt_scope_check.json')
verified_data = json.load(f)
f.close()

# Collect the organized file after step 1 of 2-step OOS verification
f = open("organized_h_n.json")
data = json.load(f)
f.close()

hard_negative = []
h_n_ins = []
ins = data['ins']
h_n_per_intent = data["statistics"]["hard negatives for each intent"]
no_generation_count = data["statistics"]["no generation count"]

# checking to see if the generated hard-negative OOS utterances pass the step 2 of the 2-step OOS verfication
for item in verified_data:
    sentence = item[0]
    intent = item[1]
    if item[2] == 'no':
        hard_negative.append([sentence, intent])
    else:
        h_n_ins.append([sentence, intent])
        h_n_per_intent[intent] -= 1


h_n_per_intent = dict(sorted(h_n_per_intent.items(), key=lambda x:x[1]))

statistics = {"intents": data["statistics"]["intents"],
              "total count": data["statistics"]["total count"],
              "hard negative oos count": len(hard_negative),
              "hard negative in scope count": len(h_n_ins),
              "all in scope data count": len(h_n_ins)+len(ins),
              "no generation count": int(data["statistics"]["total count"]) - len(hard_negative) - len(h_n_ins) - len(ins),
              "oos hard negative rate": len(hard_negative)/int(data["statistics"]["total count"]),
              "oos hard negative for each intent": h_n_per_intent
              }

output = {'h_n_oos':hard_negative, 'h_n_is':h_n_ins, 'is':ins, 'statistics': statistics}

# write the final results for hard-negative OOS in a file
with open("h_n_oos.json, 'w') as f:
    json.dump(output, f, indent=4)
