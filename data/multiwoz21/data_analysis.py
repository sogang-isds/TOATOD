import os, json, copy, re, zipfile, shutil, urllib
from urllib import request
from zipfile import ZipFile
from io import BytesIO
from collections import OrderedDict
from utils.ontology import all_domains

# 2.1
data_path = '../multiwoz21/'
save_path = 'multi-woz-analysis/'
save_path_exp = 'multi-woz-processed/'
data_file = 'data.json'
domains = all_domains


def loadDataMultiWoz():
    data_url = os.path.join(data_path, 'data.json')
    dataset_url = "https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip?raw=true"
    download_path = data_path

    if not os.path.exists(os.path.join(data_url)):
        print("Downloading and unzipping the MultiWOZ dataset")
        resp = urllib.request.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall(download_path)
        zip_ref.close()

        extract_path = os.path.join(download_path, 'MultiWOZ_2.1')
        moveFilestoDB(src_path=extract_path, dst_path= 'db/')
        moveFiles(src_path=extract_path, dst_path= data_path)
        return

def moveFilestoDB(src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    shutil.copy(os.path.join(src_path, 'attraction_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'attraction_db.json'))
    shutil.copy(os.path.join(src_path, 'hospital_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'hospital_db.json'))
    shutil.copy(os.path.join(src_path, 'hotel_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'hotel_db.json'))
    shutil.copy(os.path.join(src_path, 'police_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'police_db.json'))
    shutil.copy(os.path.join(src_path, 'restaurant_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'restaurant_db.json'))
    shutil.copy(os.path.join(src_path, 'taxi_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'taxi_db.json'))
    shutil.copy(os.path.join(src_path, 'train_db.json'), dst_path)
    os.remove(os.path.join(src_path, 'train_db.json'))
    return

def moveFiles(src_path, dst_path):
    shutil.copy(os.path.join(src_path, 'data.json'), dst_path)
    os.remove(os.path.join(src_path, 'data.json'))
    shutil.copy(os.path.join(src_path, 'slot_descriptions.json'), dst_path)
    os.remove(os.path.join(src_path, 'slot_descriptions.json'))
    shutil.copy(os.path.join(src_path, 'system_acts.json'), dst_path)
    os.remove(os.path.join(src_path, 'system_acts.json'))
    shutil.copy(os.path.join(src_path, 'testListFile.txt'), dst_path)
    os.remove(os.path.join(src_path, 'testListFile.txt'))
    shutil.copy(os.path.join(src_path, 'valListFile.txt'), dst_path)
    os.remove(os.path.join(src_path, 'valListFile.txt'))
    shutil.rmtree(os.path.join(src_path))
    shutil.rmtree('__MACOSX')
    return

def analysis():
    compressed_raw_data = {}
    goal_of_dials = {}
    req_slots = {}
    info_slots = {}
    dom_count = {}
    dom_fnlist = {}
    all_domain_specific_slots = set()
    for domain in domains:
        req_slots[domain] = []
        info_slots[domain] = []

    archive = zipfile.ZipFile(data_path+data_file+'.zip', 'w')
    archive.write(data_path+data_file)
    archive.close()

    archive = zipfile.ZipFile(data_path + data_file + '.zip', 'r')
    data = archive.open(data_path + data_file, 'r').read().decode('utf-8').lower()
    ref_nos = list(set(re.findall(r'\"reference\"\: \"(\w+)\"', data)))

    with open(data_path + data_file, 'r') as f:
        data = json.load(f)

    for fn, dial in data.items():
        goals = dial['goal']
        logs = dial['log']

        # get compressed_raw_data and goal_of_dials
        compressed_raw_data[fn] = {'goal': {}, 'log': []}
        goal_of_dials[fn] = {}
        for dom, goal in goals.items(): # get goal of domains that are in demmand
            if dom != 'topic' and dom != 'message' and goal:
                compressed_raw_data[fn]['goal'][dom] = goal
                goal_of_dials[fn][dom] = goal

        for turn in logs:
            if not turn['metadata']: # user's turn
                compressed_raw_data[fn]['log'].append({'text': turn['text']})
            else: # system's turn
                meta = turn['metadata']
                turn_dict = {'text': turn['text'], 'metadata': {}}
                for dom, book_semi in meta.items(): # for every domain, sys updates "book" and "semi"
                    book, semi = book_semi['book'], book_semi['semi']
                    record = False
                    for slot, value in book.items(): # record indicates non-empty-book domain
                        if value not in ['', []]:
                            record = True
                    if record:
                        turn_dict['metadata'][dom] = {}
                        turn_dict['metadata'][dom]['book'] = book # add that domain's book
                    record = False
                    for slot, value in semi.items(): # here record indicates non-empty-semi domain
                        if value not in ['', []]:
                            record = True
                            break
                    if record:
                        for s, v in copy.deepcopy(semi).items():
                            if v == 'not mentioned':
                                del semi[s]
                        if not turn_dict['metadata'].get(dom):
                            turn_dict['metadata'][dom] = {}
                        turn_dict['metadata'][dom]['semi'] = semi # add that domain's semi
                compressed_raw_data[fn]['log'].append(turn_dict) # add to log the compressed turn_dict


            # get domain statistics
            dial_type = 'multi' if 'mul' in fn or 'MUL' in fn else 'single' # determine the dialog's type: sinle or multi
            if fn in ['pmul2756.json', 'pmul4958.json', 'pmul3599.json']:
                dial_type = 'single'
            dial_domains = [dom for dom in domains if goals[dom]] # domains that are in demmand
            dom_str = ''
            for dom in dial_domains:
                if not dom_count.get(dom+'_'+dial_type): # count each domain type, with single or multi considered
                    dom_count[dom+'_'+dial_type] = 1
                else:
                    dom_count[dom+'_'+dial_type] += 1
                if not dom_fnlist.get(dom+'_'+dial_type): # keep track the file number of each domain type
                    dom_fnlist[dom+'_'+dial_type] = [fn]
                else:
                    dom_fnlist[dom+'_'+dial_type].append(fn)
                dom_str += '%s_'%dom
            dom_str = dom_str[:-1] # substract the last char in dom_str
            if dial_type=='multi': # count multi-domains
                if not dom_count.get(dom_str):
                    dom_count[dom_str] = 1
                else:
                    dom_count[dom_str] += 1
                if not dom_fnlist.get(dom_str):
                    dom_fnlist[dom_str] = [fn]
                else:
                    dom_fnlist[dom_str].append(fn)
            ######

            # get informable and requestable slots statistics
            for domain in domains:
                info_ss = goals[domain].get('info', {})
                book_ss = goals[domain].get('book', {})
                req_ss = goals[domain].get('reqt', {})
                for info_s in info_ss:
                    all_domain_specific_slots.add(domain+'-'+info_s)
                    if info_s not in info_slots[domain]:
                        info_slots[domain]+= [info_s]
                for book_s in book_ss:
                    if 'book_' + book_s not in info_slots[domain] and book_s not in ['invalid', 'pre_invalid']:
                        all_domain_specific_slots.add(domain+'-'+book_s)
                        info_slots[domain]+= ['book_' + book_s]
                for req_s in req_ss:
                    if req_s not in req_slots[domain]:
                        req_slots[domain]+= [req_s]


    # result statistics
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_exp):
        os.mkdir(save_path_exp)
    with open(save_path+'req_slots.json', 'w') as sf:
        json.dump(req_slots,sf,indent=2)
    with open(save_path+'info_slots.json', 'w') as sf:
        json.dump(info_slots,sf,indent=2)
    with open(save_path+'all_domain_specific_info_slots.json', 'w') as sf:
        json.dump(list(all_domain_specific_slots),sf,indent=2)
        print("slot num:", len(list(all_domain_specific_slots)))
    with open(save_path+'goal_of_each_dials.json', 'w') as sf:
        json.dump(goal_of_dials, sf, indent=2)
    with open(save_path+'compressed_data.json', 'w') as sf:
        json.dump(compressed_raw_data, sf, indent=2)
    with open(save_path + 'domain_count.json', 'w') as sf:
        single_count = [d for d in dom_count.items() if 'single' in d[0]]
        multi_count = [d for d in dom_count.items() if 'multi' in d[0]]
        other_count = [d for d in dom_count.items() if 'multi' not in d[0] and 'single' not in d[0]]
        dom_count_od = OrderedDict(single_count+multi_count+other_count)
        json.dump(dom_count_od, sf, indent=2)
    with open(save_path_exp + 'reference_no.json', 'w') as sf:
        json.dump(ref_nos,sf,indent=2)
    with open(save_path_exp + 'domain_files.json', 'w') as sf:
        json.dump(dom_fnlist, sf, indent=2)


if __name__ == '__main__':
    loadDataMultiWoz()
    analysis()