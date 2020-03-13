import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from bokeh.plotting import figure
from bokeh.io import show
from geopy.geocoders import Nominatim                       
from pytrends.request import TrendReq
pytrend = TrendReq()

geolocator = Nominatim(user_agent="Pytorch Disaster Classifier")
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
nlp = en_core_web_sm.load()
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import logging
# #from sklearn.model_selection import StratifiedKFold
# #from sklearn.metrics import accuracy_score, f1_score

#from transformers import *

from nltk.tokenize import word_tokenize

import os
import re
import string
import random
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tweet=st.text_area("Enter Tweet","Enter text")

data = {'id':[0],'text':[tweet]} 
  
# Create DataFrame 
df = pd.DataFrame(data) 

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())

fh = logging.FileHandler('log_model.txt')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


# from sklearn.externals import joblib
# model=joblib.load('model.pkl')

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, text, label=None):
        """Constructs a InputExample.
        Args:
            id: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        
        self.example_id = example_id
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        self.label = label


class NeuralNet(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(NeuralNet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.drop_out = nn.Dropout() # dropout layer to prevent overfitting
        self.fc = nn.Linear(hidden_size, num_classes) # fully connected layer
        
    def forward(self, input_ids, input_mask, segment_ids):
        last_hidden_state, pooler_output, all_hidden_states, all_attentions = self.bert(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
        last_hidden_state = last_hidden_state[:, 0,:]                                                       
        
        # Linear layer expects a tensor of size [batch size, input size]
        out = self.drop_out(last_hidden_state) 
        out = self.fc(out) 
        return F.log_softmax(out)

model=NeuralNet()
#model = torch.load('model.pkl',map_location=torch.device('cpu') )
model.load_state_dict(torch.load('model.pkl', map_location='cpu'))
model.eval() 


# Hyperparameters

MAX_SEQ_LENGTH = 512  
LEARNING_RATE = 1e-5  
NUM_EPOCHS = 3  
BATCH_SIZE = 8  
PATIENCE = 2  
FILE_NAME = 'model' 
NUM_FOLDS = 5

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    
    features = []
    
    for example_index, example in enumerate(examples):
        
        text = tokenizer.tokenize(example.text)
        MAX_TEXT_LEN = max_seq_length - 2 
        text = text[:MAX_TEXT_LEN]

        choices_features = []

        tokens = ["[CLS]"] + text + ["[SEP]"]  
        segment_ids = [0] * (len(text) + 2) 
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("id: {}".format(example.id))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(len(input_mask)))
            logger.info("segment_ids: {}".format(len(segment_ids)))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.id,
                choices_features=choices_features,
                label=label
            )
        )
    return features

def read_examples(df, is_training):
    if not is_training:
        df['target'] = np.zeros(len(df), dtype=np.int64)
    examples = []
    for val in df[['id', 'text', 'target']].values:
        examples.append(InputExample(id=val[0], text=val[1], label=val[2]))
    return examples, df

def select_field(features, field):
    return [feature.choices_features[field] for feature in features]
    
test_examples, test_df = read_examples(df, is_training=False)
test_features = convert_examples_to_features(test_examples, tokenizer, MAX_SEQ_LENGTH, True)
test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)

test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)

test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

#Prediction
prediction=model(test_input_ids,test_input_mask, test_segment_ids)

st.markdown(prediction)

#st.markdown(model.eval())
class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            return key
            #print(key + ' - ' + str(value))
            if i > number:
                break
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight
        

st.title("Disaster Tweets Classifier")


location_raw=st.text_input("Location", "Enter Location")

#using spacy keyword extraction
tr4w = TextRank4Keyword()
tr4w.analyze(tweet, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
#Change the number to get more keywords
keyword=tr4w.get_keywords(-2)



term=keyword+"+"+location_raw
tweet_fake=True


if st.button("Submit"):
    if tweet_fake==False:
        st.warning("The Tweet is fake")
    elif tweet_fake==True:
        st.success("The Tweet is real")

        pytrend.build_payload(kw_list=[term])# Interest by Region


   
        interest_over_time_df = pytrend.interest_over_time()
        interest_over_time_df['date'] = interest_over_time_df.index
        interest_over_time_df['date'] = pd.to_datetime(interest_over_time_df['date'])
                
        fig=figure(x_axis_type='datetime',plot_width=600,plot_height=300,y_axis_label='Search Popularity Index',title="Google Search Popularity")
        fig.line(interest_over_time_df['date'],interest_over_time_df[term],
        color='red',legend=term)
        st.bokeh_chart(fig)


        location = geolocator.geocode(location_raw)
        location = geolocator.reverse([location.latitude, location.longitude])
        df = pd.DataFrame({'lat':[location.latitude], 'lon':[location.longitude]},columns=['lat', 'lon'])
        st.map(df,zoom=10)



models = []

models.append(model)
preds = []

for i, model in enumerate(models):
    
    test_preds = []
    
    model.eval()

    with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = tuple(t.cuda() for t in batch)
                x_ids, x_mask, x_sids = batch
                y_pred = model(x_ids, x_mask, x_sids).detach()
                test_preds[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = F.softmax(y_pred, dim=1).cpu().numpy()  
    
    print("MODEL {}: inference done".format(i))
    
    preds.append(test_preds)