import os
import numpy as np
import csv
import json
twitter_train_path = '/home/ylin/MABSA/data/IJCAI2019_data/twitter2017/train.tsv'
twitter_test_path = '/home/ylin/MABSA/data/IJCAI2019_data/twitter2017/test.tsv'
twitter_dev_path = '/home/ylin/MABSA/data/IJCAI2019_data/twitter2017/dev.tsv'


def aesc(line):
    raw_sentence = line[3]
    raw_aspect = line[4]
    raw_words = raw_sentence.split()
    # sentence = raw_sentence.replace('$T$', raw_aspect)
    # words = raw_sentence.split()
    # dic['words'] = words
    a_split = raw_aspect.split()
    aspect_len = len(a_split)
    # aspects = []
    a_dic = {}
    a_dic['from'] = raw_words.index('$T$')
    a_dic['to'] = a_dic['from'] + aspect_len

    p = line[1]
    if p == '0':
        polarity = 'NEG'
    elif p == '1':
        polarity = 'NEU'
    else:
        polarity = 'POS'

    a_dic['polarity'] = polarity
    a_dic['term'] = a_split

    return a_dic


def process(path):
    with open(path) as f:
        tsvreader = csv.reader(f, delimiter='\t')
        table = []
        for line in tsvreader:
            table.append(line)
        # print(table[1])
        i = 1
        MRC_data = []
        diff = 0
        while i < len(table):
            dic = {}
            raw_sentence = table[i][3]
            raw_aspect = table[i][4]
            sentence = raw_sentence.replace('$T$', raw_aspect)
            words = sentence.split()
            dic['words'] = words
            aspects = []

            aspects.append(aesc(table[i]))

            img_id = table[i][2]
            dic['image_id'] = img_id
            while i + 1 < len(table):
                raw_sentence_tt = table[i + 1][3]
                raw_aspect_tt = table[i + 1][4]
                sentence_tt = raw_sentence_tt.replace('$T$', raw_aspect_tt)
                img_id_tt = table[i + 1][2]
                if sentence_tt == sentence:
                    i += 1
                    if not img_id_tt == img_id:
                        diff += 1
                    aspects.append(aesc(table[i]))
                else:
                    break

            dic['aspects'] = aspects
            opinions = []
            o_dic = {}
            o_dic['term'] = []
            opinions.append(o_dic)

            dic['opinions'] = opinions

            MRC_data.append(dic)
            i += 1
        print('diff', diff)
        return MRC_data


test_data = process(twitter_test_path)
print('ok!!!')
test_json = json.dumps(test_data)
f_test = open('twitter2017/test.json', 'w')
f_test.write(test_json)
f_test.close()

dev_data = process(twitter_dev_path)
dev_json = json.dumps(dev_data)
f_dev = open('twitter2017/dev.json', 'w')
f_dev.write(dev_json)
f_dev.close()

train_data = process(twitter_train_path)
train_json = json.dumps(train_data)
f_train = open('twitter2017/train.json', 'w')
f_train.write(train_json)
f_train.close()
