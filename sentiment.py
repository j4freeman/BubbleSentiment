from __future__ import print_function, division
import sys

from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

OUTPUT_PATH = 'test_sentences.csv'

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import mysql.connector as sql

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


cnx = sql.connect(host='mysql.eecs.ku.edu', user='dfernand', password='BubbleUp582', database='dfernand')
cursor = cnx.cursor()

query = ("SELECT id, body, content_type FROM Posts WHERE content_type < 536870912;")

cursor.execute(query)

output = []

for row in cursor:
  output.append(list(row))

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

    # try:
    #     print(output[i])
    #     decode_int(output[i][-1])
    # except Exception as e:
    #     print("Exception at row {}!".format(i))
    #     print(str(e))


for i in output:
    query = "UPDATE Sentiment_demo SET content_type = %s WHERE id = %s;"
    cursor.execute(query, (i[-1], i[0]))

cnx.commit()
cursor.close()
cnx.close()
