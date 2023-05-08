import json
import os

file = os.listdir()

with open(f'./result2.json','r') as fp:
    results = json.load(fp)

changed = {}
for session in results:
    k = session['dial_id']
    if k not in changed:
        changed[k] = []
    changed[k].append({'response':session['resp_gen']})

with open(f'./result3.json','w') as fp:
    json.dump(changed,fp)