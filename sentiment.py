from __future__ import print_function, division
import sys

import json
import numpy as np
from DM.deepmoji.sentence_tokenizer import SentenceTokenizer
from DM.deepmoji.model_def import deepmoji_emojis
from DM.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import requests

def discretize_sent(snt):
    if snt == 0.0:
        return 3
    if snt > 0.67:
        return 6
    if snt > 0.33:
        return 5
    if snt > 0.0:
        return 4
    if snt > -0.33:
        return 2
    if snt > -0.67:
        return 1
    return 0

def convert_emoji_bin(e):
    if e == -1:
        return str(format(0, '06b'))
    return str(format(e, '06b'))

def convert_vals(in_list):
    in_list[5] = discretize_sent(in_list[5])
    tempstr = "001"
    tempstr += str(format(in_list[0], '05b'))
    tempstr += str(format(in_list[4], '03b'))
    tempstr += convert_emoji_bin(in_list[1])
    tempstr += convert_emoji_bin(in_list[2])
    tempstr += convert_emoji_bin(in_list[3])
    tempstr += str(format(in_list[5], '03b'))
    return int(tempstr, 2)

def translate_sentiment(snt):
    if snt == 3:
        return 'neutral'
    if snt == 0:
        return 'strong neg'
    if snt == 1:
        return 'med neg'
    if snt == 2:
        return 'weak neg'
    if snt == 4:
        return 'weak pos'
    if snt == 5:
        return 'med pos'
    if snt == 6:
        return 'strong pos'

def decode_int(val):
    tempstr = str(format(val, '32b'))
    analyzed = int(tempstr[:3], 2)
    type = int(tempstr[3:8], 2)
    num = int(tempstr[8:11], 2)
    e1 = int(tempstr[11:17], 2)
    e2 = int(tempstr[17:23], 2)
    e3 = int(tempstr[23:29], 2)
    snt = int(tempstr[29:], 2)
    if num == 2:
        e3 = -1
    elif num == 1:
        e2 = -1
        e3 = -1
    elif num == 0:
        e3 = -1
        e2 = -1
        e1 = -1
    print("{}: The analyzed bit is: {}, the type is: {}, the number of emoji is: {}, the first emoji is: {}, the second emoji is: {}, the third emoji is: {}, the sentiment is: {}".format(str(val), analyzed, type, num, e1, e2, e3, translate_sentiment(snt)))


mydict = {
    "Content-Type": "application/json",
    "Authorization": "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6NH0.nc4RJ7Ni-ALRyz2LGXuUq8Eofi_hZlxF6CBBt_ziRGw"
}

r = requests.get('http://bubbleup-api.herokuapp.com/posts', headers=mydict)

if not r:
    sys.exit()

output = r.json()
output = [[x[u'id'], x[u'body'], x[u'content_type']] for x in output if x[u'content_type'] < 536870912]

if not output:
    sys.exit()

TEST_SENTENCES = [x[1] for x in output]

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 60
batch_size = 8

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

model = deepmoji_emojis(maxlen, PRETRAINED_PATH)

prob = model.predict(tokenized)

for i, t in enumerate(TEST_SENTENCES):
    t_tokens = tokenized[i]
    t_score = [t]
    t_prob = prob[i]
    ind_top = top_elements(t_prob, 3)
    ind_top = [x if t_prob[x] >= 0.1 else -1 for x in ind_top]
    output[i].extend(ind_top)
    output[i].append(sum(x > -1 for x in ind_top))

    snt = analyzer.polarity_scores(output[i][1])
    output[i].append(snt['compound'])
    out_int = convert_vals(output[i][2:])
    output[i].append(out_int)

    try:
        print("=============")
        print(output[i])
        decode_int(output[i][-1])
        print('http://bubbleup-api.herokuapp.com/posts/' + str(output[i][0]))
        print("=============")
    except Exception as e:
        print("Exception at row {}!".format(i))
        print(str(e))
    sys.stdout.flush()

# for i in output:
#     url = 'http://bubbleup-api.herokuapp.com/posts/' + i[0]
#     r = requests.put(url, headers=mydict, json={u'content_type': i[-1]})
#     if not r:
#         print("failed to post for id: ", str(i[0]))
