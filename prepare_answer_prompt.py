# prepare_answer_prompt.py
import pandas as pd
import pickle
import os 
import json

data_dir = '/mnt/storage/swapnanil_mukherjee/e-snli-ve/'
split = 'test'
annt_file = 'esnlive_{}_w_captions.csv'.format(split)
annt_df = pd.read_csv(os.path.join(data_dir, annt_file))
sample_df = annt_df.sample(100)

with open("wikidata_esnlive_topentities.pkl", 'rb') as f:
    top_entities = pickle.load(f)
with open("entity_data.pkl", 'rb') as f:
    entity_data = pickle.load(f)

# Create a list to store prompts and metadata
prompts_data = []    
for row in sample_df.itertuples(): # replace with annt_df for actual querying
    index = row[0]
    question = row.hypothesis
    context = row.caption
    image_name = row.Flickr30kID
    
    top_ids = top_entities[image_name][0]
    descriptions = []
    for idx in top_ids:
        descriptions.append(entity_data[idx])
    knowledge_items = ["is a".join(item) for item in descriptions][:10]
    
    answer_prompt = '''
Please answer if the hypothesis entails, contradicts or is neutral to the above context using the context and the provided knowledge if given.
===
Context: a woman skating on ice in a competition
===
H: an olympic hopeful prepares for the big day
A: neutral 
===
.....
===
Context: a boy with a mask and snorkel in a tub 
===
H: A boy is scuba diving at home.
A: entailment 
===
.....
===
Context: a keyboard sitting in front of the 
===
H: a soccer player lying on the ground
A: contradiction 
===
.....
===
Context: {}
Knowledge:\n{}
===
H: {}
A:
'''
    answer_prompt = answer_prompt.format(context, "\n".join(knowledge_items), question)
    
    prompt_data = {
        "index": index,
        "image_name": image_name,
        "prompt": answer_prompt
    }
    prompts_data.append(prompt_data)

# Save prompts 
with open("gpt_batches/esnlive_sample_prompts.jsonl", 'w') as f:
    for sample in prompts_data:
        f.write(json.dumps(sample)+'\n')

print("JSON file written.")