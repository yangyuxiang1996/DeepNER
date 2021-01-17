'''
处理传统的序列标注data为json格式
'''
import json
import os
from tqdm import trange
from sklearn.model_selection import KFold, train_test_split


def save_info(datadir, data, type):
    with open(os.path.join(datadir, f"{type}.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def convert_data_to_json(datadir, type, split=False, save_data=False, save_dict=False):
    examples = []
    labels = []
    sentence = []
    n = 0
    i = 0
    entity = ""
    first_start = 0
    first_end = 0
    
    with open(os.path.join(datadir, f"{type}.txt"), "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip()
            if line != '':
                line = line.split("\t")
                assert len(line) == 2
                sentence.append(line[0])
                ann = line[1]
                if ann != "O":
                    head, type = ann.split('-')[0], ann.split('-')[1]
                    assert type in ["BRAND", "TYPE", "PRICE", "OUTLOOK", "SYS", "HARD", "FUNC", "SCENE"]
                    if head == "B":
                        first_end = i
                        if entity and first_end > first_start:
                            labels.append([f"T{len(labels)+1}", type, first_start, first_end, entity])
                        first_start = first_end
                        entity = line[0]
                    elif head == "I":
                        entity += line[0]
                else:
                    first_end = i
                    if entity and first_end > first_start:
                        labels.append([f"T{len(labels)+1}", type, first_start, first_end, entity])
                        entity = ""

                i += 1
            else:
                if labels:
                    examples.append({'id': n,
                               'text': "".join(sentence),
                               'labels': labels,
                               'pseudo': 0})
                    labels = []
                    sentence = []
                    n += 1
                    i = 0

    # 构建实体知识库
    kf = KFold(10)
    entities = set()
    ent_types = set()
    for _now_id, _candidate_id in kf.split(examples):
        now = [examples[_id] for _id in _now_id]
        candidate = [examples[_id] for _id in _candidate_id]
        now_entities = set()

        for _ex in now:
            for _label in _ex['labels']:
                ent_types.add(_label[1])

                if len(_label[-1]) > 1:
                    now_entities.add(_label[-1])
                    entities.add(_label[-1])
        # print(len(now_entities))
        for _ex in candidate:
            text = _ex['text']
            candidate_entities = []

            for _ent in now_entities:
                if _ent in text:
                    candidate_entities.append(_ent)

            _ex['candidate_entities'] = candidate_entities
    assert len(ent_types) == 8

    raw_data_dir = os.path.join(datadir, 'raw_data')
    if split:
        train_examples, dev_examples = train_test_split(examples, random_state=2021, shuffle=True, test_size=0.2)
        if save_data:
            save_info(raw_data_dir, train_examples, "train")
            save_info(raw_data_dir, dev_examples, "dev")
    else:
        if save_data:
            save_info(raw_data_dir, type, examples)

    if save_dict:
        ent_types = list(ent_types)
        span_ent2id = {_type: i+1 for i, _type in enumerate(ent_types)}

        ent_types = ['O'] + [p + '-' + _type for p in ['B', 'I'] for _type in list(ent_types)]
        crf_ent2id = {ent: i for i, ent in enumerate(ent_types)}

        mid_data_dir = os.path.join(datadir, 'mid_data')
        if not os.path.exists(mid_data_dir):
            os.mkdir(mid_data_dir)

        save_info(mid_data_dir, span_ent2id, 'span_ent2id')
        save_info(mid_data_dir, crf_ent2id, 'crf_ent2id')


def build_ent2query(data_dir):
    # 利用比赛实体类型简介来描述 query
    ent2query = {
        # 药物
        'DRUG': "找出药物：用于预防、治疗、诊断疾病并具有康复与保健作用的物质。",
        # 药物成分
        'DRUG_INGREDIENT': "找出药物成分：中药组成成分，指中药复方中所含有的所有与该复方临床应用目的密切相关的药理活性成分。",
        # 疾病
        'DISEASE': "找出疾病：指人体在一定原因的损害性作用下，因自稳调节紊乱而发生的异常生命活动过程，会影响生物体的部分或是所有器官。",
        # 症状
        'SYMPTOM': "找出症状：指疾病过程中机体内的一系列机能、代谢和形态结构异常变化所引起的病人主观上的异常感觉或某些客观病态改变。",
        # 症候
        'SYNDROME': "找出症候：概括为一系列有相互关联的症状总称，是指不同症状和体征的综合表现。",
        # 疾病分组
        'DISEASE_GROUP': "找出疾病分组：疾病涉及有人体组织部位的疾病名称的统称概念，非某项具体医学疾病。",
        # 食物
        'FOOD': "找出食物：指能够满足机体正常生理和生化能量需求，并能延续正常寿命的物质。",
        # 食物分组
        'FOOD_GROUP': "找出食物分组：中医中饮食养生中，将食物分为寒热温凉四性，"
                      "同时中医药禁忌中对于具有某类共同属性食物的统称，记为食物分组。",
        # 人群
        'PERSON_GROUP': "找出人群：中医药的适用及禁忌范围内相关特定人群。",
        # 药品分组
        'DRUG_GROUP': "找出药品分组：具有某一类共同属性的药品类统称概念，非某项具体药品名。例子：止咳药、退烧药",
        # 药物剂量
        'DRUG_DOSAGE': "找出药物剂量：药物在供给临床使用前，均必须制成适合于医疗和预防应用的形式，成为药物剂型。",
        # 药物性味
        'DRUG_TASTE': "找出药物性味：药品的性质和气味。例子：味甘、酸涩、气凉。",
        # 中药功效
        'DRUG_EFFICACY': "找出中药功效：药品的主治功能和效果的统称。例子：滋阴补肾、去瘀生新、活血化瘀"
    }

    with open(os.path.join(data_dir, 'mrc_ent2id.json'), 'w', encoding='utf-8') as f:
        json.dump(ent2query, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import os
    print(os.getcwd())
    convert_data_to_json('data/weibo', "stack", split=True, save_data=True, save_dict=True)
    build_ent2query('data/weibo/mid_data')

        
                

