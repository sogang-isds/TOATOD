import os
import pandas as pd
import json
import requests

data_link = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"

def DownloadClinc150():
    dataset_url = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"

    if not os.path.exists('data_full.json'):
        print("Downloading the Clinc150 dataset")
        r = requests.get(dataset_url, allow_redirects=True)
        open('data_full.json', 'wb').write(r.content)
        print("Downloaded the Clinc150 dataset")
        return

def ProcessClinc150():
    print("Processing the Clinc150 dataset")
    with open('data_full.json') as f:
        data = json.load(f)

    print(data.keys())

    text = []
    label = []
    full_label = []
    for i in ('train', 'val', 'test'):
        for d in data[i]:
            if d[1] != 'oov':
                text.append(d[0])
                label.append(d[1])
                full_label.append(d[1])

        df = pd.DataFrame({'text': text, 'category': label})
        df.to_csv('{}.csv'.format(i), index=False)
        print("num of {} is {}".format(i, len(text)))

    print("num of in-scope intent classesse: ", len(set(full_label)))

if __name__ == '__main__':
    DownloadClinc150()
    ProcessClinc150()
