import numpy as np
import pandas as pd
import ast
import glob
import os
import re
import emoji

from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.tokenize import wordpunct_tokenize
from scipy.linalg import triu
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class chat_filter:

    def __init__(self):

        self.chat_path = './data/chatlogs'
        self.emote_path = './data/emotes/other_emotes'

        self.emote_data = []
        self.emote_libs()
        self.master_df = pd.DataFrame()
        self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

    # ----------------------------------------------------------------------------------------------
    # Loading and doing initial cleaning
    def load_files(self, file_name):

        # read in file
        col_force = {
            'message': pd.StringDtype(),
            'emotes': object,
            'emotes_id': object,
            'username': pd.StringDtype()
        }
        
        df = pd.read_csv(file_name, dtype=col_force, encoding='utf-8')
        df['message'] = df['message'].str.rstrip(" \t\n\r\f\v\u00A0\U000E0000")
        df['message'] = df['message'].apply(self.remove_at)
        df['emotes'] = df['emotes'].apply(ast.literal_eval)
        df['message'] = df['message'].apply(emoji.demojize)

        # check if there are 3rd party emotes
        df['emotes'] = df.apply(lambda row: self.has_third_party(row['message'], row['emotes']), axis=1)

        # if no emotes exist remove it
        df = df[df['emotes'].apply(lambda x: x != [])]

        # filter out emote only messages
        df['is_emote_only'] = df.apply(self.is_emote_only, axis=1)
        df = df[~df['is_emote_only']]

        # finalize and create the message without the emote in one column
        df['clean_message'] = df.apply(lambda row: self.emote_removal(row['message'], row['emotes']), axis=1)

        # use label maker here to create the associated bucket emotion label
        df = self.label_maker(df)
        # finalize and save the data to a new master dataframe and output as a csv
        print(f"Length of entries gathered: {df.shape[0]}")

        return df

    # [HELPER FUNCTION] for removing mentions
    def remove_at(self, message):
        return re.sub(r'@\w+', '', message)

    # [HELPER FUNCTION] for identifying 3rd party emotes
    def has_third_party(self, text, current_emotes):
        # has third party from emote data
        pattern = r'\b(' + '|'.join(re.escape(emote) for emote in self.emote_data) + r')\b'
        matches = re.findall(pattern, text)
        current_emotes.extend(matches)

        # has third party from unicode
        unicode_emoji = re.findall(r':(.*?):', text)
        emote_colons = [f":{e}:" for e in unicode_emoji]
        current_emotes.extend(emote_colons)

        return current_emotes 

    # [HELPER FUNCTION] for filter layer for detecting emote only
    def is_emote_only(self, row):

        message = row['message'].strip()
        emotes_list = row['emotes']
        
        if emotes_list:
            for emote in emotes_list:
                message = message.replace(emote, "")
        
        message = message.rstrip(" \t\n\r\f\v\u00A0\U000E0000")
        return message == ""
    
    # [HELPER FUNCTION] to remove all emotes from a message for label creating
    def emote_removal(self, message, emotes):
        return ' '.join([word for word in message.split() if word not in emotes])
    
    # [HELPER FUNCTION] to detect 
    # According to other emote libraries that are 3rd party:
    # Loads top 100 Emotes from 7tv, BTTV, frankerzface, regular emojis - identify and put in df
    def emote_libs(self):
        
        for emote_list in glob.glob(os.path.join(self.emote_path, '*.txt')):
            print(f"Adding {emote_list} to emote library.")

            with open(emote_list, 'r') as file:
                temp = file.read()
                temp_emotes = temp.split(",")
                self.emote_data.extend(temp_emotes)

        print("External Emotes Loaded.")

    # ----------------------------------------------------------------------------------------------
    # Utilizes word2vec and glove to process all of the emotes into one of the 7 categories and label
    def label_maker(self, df):
        ekman = ['anger', 'disgust', 'fear', 'hapiness', 'sadness', 'surprise', 'neutral']
        # using spacy static context
        #w2v_mod = spacy.load("en_core_web_md")
        #ek_vect = {e: w2v_mod.vocab[e].vector for e in ekman}

        # using BERT dynamic context
        bert_mod = SentenceTransformer('all-MiniLM-L6-v2')
        bert_mod = bert_mod.to('cuda')
        ek_vect = {e: bert_mod.encode(e) for e in ekman}

        # Using spacy
        # 1. emote split into subwords if possible
        #df['tokens'] = df['emotes'].apply(self.split_known)
        # 2. word2vec + glove
        #df['vector'] = df['tokens'].apply(lambda t: self.split_vectors(t, w2v_mod))
        # 3. cosine similarity to 6/7 ekman categories
        #df['assigned_emotion'] = df['vector'].apply(lambda v: self.cosine_similarity(v, ek_vect))

        # Using BERT
        df['vector'] = df['message'].apply(lambda msg: bert_mod.encode(msg))
        
        # Apply cosine similarity and extract both weights + label
        emotion_results = df['vector'].apply(lambda v: self.bert_cosine_similarity(v, ek_vect))
        
        # Create a new column for the full weights dictionary
        df['bert_weights'] = emotion_results.apply(lambda res: res['weights'])

        # Create a separate column for just the top emotion label
        df['bert_emotion'] = emotion_results.apply(lambda res: res['emotion'])

        # zershot method
        df['zeroshot_result'] = df['emotes'].apply(self.zero_shot_emotes)
        df['zeroshot_emotion'] = df['zeroshot_result'].apply(lambda x: x['emotion'])
        df['zeroshot_weights'] = df['zeroshot_result'].apply(lambda x: x['weights'])
        df['summed_weights'] = df.apply(self.summed_weights, axis=1)
        df['assigned_emotion'] = df['summed_weights'].apply(lambda w: max(w, key=w.get))
        
        # apply your own column with overrides
        df['override_flag'] = 0
        df['override_emotion'] = ''
        
        return df

    # [HELPER FUNCTION] [Spacy] Helps with splitting words into known entities for tokenization
    def split_known(self, emote):

        if isinstance(emote, list):
            tokens = []
            for e in emote:
                tokens.extend(self.split_known(e))  # recursive call for each string
            return tokens
        elif isinstance(emote, str):
            extract = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', emote)
            if not extract:
                extract = wordpunct_tokenize(emote)
            return [x.lower() for x in extract if x.isalpha()]
        else:
            return []
    
    # [HELPER FUNCTION] [Spacy] vectorization
    def split_vectors(self, parts, model):
        temp_vect = []
        for p in parts:
            lex = model.vocab[p]
            if lex.has_vector:
                temp_vect.append(lex.vector)

        if temp_vect:
            return np.mean(temp_vect, axis=0)
        else:
            return np.zeros(model.vocab.vectors_length)
        
    # [HELPER FUNCTION] [Spacy] cosine similarity calculations
    def cosine_similarity(self, v, ekv):
        if np.linalg.norm(v) == 0:
            return 'unknown'
      
        similarities = {emo: cosine_similarity([v], [vec])[0][0] for emo, vec in ekv.items()}
        return max(similarities, key=similarities.get)      

    # [HELPER FUNCTION] [BERT] vectorization and weights
    def bert_cosine_similarity(self, v, ek_vecs):
        if np.linalg.norm(v) == 0:
            return {
                'emotion': 'unknown',
                'weights': {e: 0.0 for e in ek_vecs}
            }
        
        similarities = {
            emo: cosine_similarity([v], [vec])[0][0]
            for emo, vec in ek_vecs.items()
        }
        top_emotion = max(similarities, key=similarities.get)

        return {
            'emotion': top_emotion,
            'weights': similarities
        }
    
    # [HELPER FUNCTION] [ZeroShot] vectorization and weights
    def zero_shot_emotes(self, emote_list):
        candidate_labels = ["anger", "disgust", "fear", "hapiness", "sadness", "surprise", "neutral"]

        if not emote_list or not isinstance(emote_list, list):
            return {'emotion': 'unknown', 'weights': {label: 0.0 for label in candidate_labels}}

        clean_emotes = [e for e in emote_list if isinstance(e, str) and e.strip()]
        if not clean_emotes:
            return {'emotion': 'unknown', 'weights': {label: 0.0 for label in candidate_labels}}

        emote = clean_emotes[0]  # You can adapt this to average over multiple

        try:
            result = self.zero_shot(emote, candidate_labels)
            labels = result["labels"]
            scores = result["scores"]
            weights = dict(zip(labels, scores))
            weights_full = {label: weights.get(label, 0.0) for label in candidate_labels}
            top_emotion = labels[0]

            return {
                'emotion': top_emotion,
                'weights': weights_full
            }
        
        except Exception as e:

            print(f"Error during zero-shot: {e}")
            return {'emotion': 'unknown', 'weights': {label: 0.0 for label in candidate_labels}}
    
    # [HELPER FUNCTION] Summing weights from multiple columns
    def summed_weights(self, row):
    # Sum values in both dicts by key
        return {
            key: row['bert_weights'].get(key, 0.0) + row['zeroshot_weights'].get(key, 0.0)
            for key in row['bert_weights']
        }
    
    # ----------------------------------------------------------------------------------------------
    # Batch processes all chatlogs and returns it to master_df and writes locally to
    def batch_chat(self, chat):

        print("-------\nProcessing Begins:")
        for log in os.listdir(self.chat_path):
            print(f"Working on: {log}")
            curr_file = os.path.join(self.chat_path, log)

            if os.path.isfile(curr_file):

                try:
                    df = pd.read_csv(curr_file, encoding='utf-8')
                    print(f"Successfully loaded {curr_file}")
                    loaded = True
                except UnicodeDecodeError:
                    print(f"Failed to load {curr_file}")
                    break

                temp_df = chat.load_files(curr_file)
                self.master_df = pd.concat([self.master_df, temp_df], ignore_index=True)


        print("-------\nSaving to master file.")
        output_file = './datra/official_dataset/master_chat_v3.csv'
        self.master_df.to_csv(output_file, index=False)
        print("Process Complete.")
                

    # have predone emotes using the word2vec and glove method to make things faster for the 100 emotes
    # if unencountered have label maker do it live.
    def emote_database(self):
        pass


if __name__ == "__main__":
    
    chat = chat_filter()

    # Testing
    #chat.load_files('2025-04-16-ironmouse-01.csv')
    #chat.load_files('2025-04-16-GEEGA-01.csv')

    # Actual Run for Multiple streams from chatlogs directory
    chat.batch_chat(chat)
