# -*- coding: utf-8 -*-

""" Use DeepMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the DeepMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division
import example_helper
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

OUTPUT_PATH = 'test_sentences.csv'

TEST_SENTENCES = [
u"Skynet, soon.",
u"my second bubble",
u"Second test.",
u"Coding at the reserve!",
u"wifi is sucking today, wtf",
u"❤️",
u"my post",
u"Ms Hannah here",
u"Hogar dulce hogar.",
u"What happens here stays here!",
u"I am being chased by a killer robot! Help!",
u"trying to study",
u"From Helsinki with love!",
u"https://www.linux.org",
u"Awesome song! https://youtu.be/KmlZ1WhlYsM",
u"10 4",
u"He is sleeping",
u"I'll be back!!",
u"You will see this",
u"New bubble",
u"Pic Test",
u"Hola Mundo!",
u"When it happens... https://youtu.be/pFptt7Cargc",
u"Free pizza over here!",
u"Sharing is caring",
u"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor i",
u"El Ultimo Amanecer! https://youtu.be/5_-vpQmVDMk",
u"Arriba Arriba!!",
u"Land of free health care and maple syrup",
u"GNU + Linux https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Gnu-and-penguin-color.png/220px-Gnu-and-penguin-color.png"
]

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


maxlen = 60
batch_size = 8

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

print('Running predictions.')
prob = model.predict(tokenized)

# Find top emojis for each sentence. Emoji ids (0-63)
# correspond to the mapping in emoji_overview.png
# at the root of the DeepMoji repo.
print('Writing results to {}'.format(OUTPUT_PATH))
scores = []
for i, t in enumerate(TEST_SENTENCES):
    t_tokens = tokenized[i]
    t_score = [t]
    t_prob = prob[i]
    ind_top = top_elements(t_prob, 5)
    t_score.append(sum(t_prob[ind_top]))
    t_score.extend(ind_top)
    t_score.extend([t_prob[ind] for ind in ind_top])
    scores.append(t_score)
    print(t_score)

with open(OUTPUT_PATH, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    writer.writerow(['Text', 'Top5%',
                     'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
                     'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
    for i, row in enumerate(scores):
        try:
            writer.writerow(row)
        except Exception as e:
            print("Exception at row {}!".format(i))
            print(str(e))
