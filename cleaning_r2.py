import numpy as np
import pandas as pd
import re
import ast
from collections import defaultdict
import string

class emote_analyzer:

    def __init__(self):
        self.master_record = "./data/official_dataset/master_chat_v3.csv"
        self.extra_records = "./data/official_dataset/master_chat_v2.csv"
        self.override_path = "./data/emotes/override/override.txt"
        self.final_path = "./data/official_dataset/model_ready_data_v2.csv"
        self.override_dict = {}

    # Helps load the data
    def data_loader(self):
        return pd.read_csv(self.master_record, encoding="utf-8")
    
    # taking into account extra text forms like links, raid messages, more special characters    
    def additional_filtering(self, df):

        df = df[~df['message'].str.contains("https://", na=False)]
        df = df[~df['message'].str.contains("𝑹𝒂𝒊𝒅", na=False)]
        df = df[df['message'].apply(lambda x: re.fullmatch(r'[\x00-\x7F]+', str(x)) is not None)]

        return df

    # Reads a txt dictionary of common emotes and their associated labels and applies that to final label
    def override_labels(self, df):
        
        # read emote column only and lowercase them
        df_copy = df.copy()
        df_copy['sd_emote'] = df['emotes'].str.lower()

        # read txt dictionary (general emotion appendages like wheeze, gun, monka, kek, o7, cry, etc.)
        with open(self.override_path, 'r', encoding='utf-8') as f:
            for entry in f:
                temp_line = entry.split(",")
                if len(temp_line) == 2:
                    override_emote = temp_line[0].strip()
                    override_cat = temp_line[1].strip()
                    self.override_dict[override_emote] = override_cat

        # override emotions
        df_copy['override_weights'] = df_copy['emotes'].apply(self.override_weight)

        # check current results
        #pd.set_option('display.max_columns', None)
        #pd.set_option('display.width', None)  
        #pd.set_option('display.max_colwidth', None)
        #print(df_copy['override_weights'])

        df_copy['summed_weights'] = df_copy['summed_weights'].apply(ast.literal_eval)

        # add to pre-existing weights if there are multiple emotes and judge.
        df_copy['final_weights'] = df_copy.apply(self.add_weights, axis=1)

        # create new column final weights + final emotion decision
        df_copy['final_emotion'] = df_copy['final_weights'].apply(lambda w: max(w, key=w.get))

        return df_copy
    
    # gives us our override weight of adding 1 to emphasize importance, same formatting of weights
    # as before with cleaning.py
    def override_weight(self, emote_str):
        candidate_labels = ["anger", "disgust", "fear", "hapiness", "sadness", "surprise", "neutral"]

        try:
            emotes = ast.literal_eval(emote_str)
        except:
            emotes = []

        emotes = [e.lower() for e in emotes]
        weights = defaultdict(int)

        for emote in emotes:
            for key in self.override_dict:
                if key in emote:
                    label = self.override_dict[key]
                    weights[label] += 1

        weights_full = {label: weights.get(label, 0) for label in candidate_labels}
        return weights_full
    
    # sums up the weights between the summed weights from the first two proceses and the override
    def add_weights(self, row):

        return {
            key: row['summed_weights'].get(key, 0.0) + row['override_weights'].get(key, 0.0)
            for key in row['summed_weights']
        }
    
    # remotes emotes to stage for training.
    def remove_emotes(self, row):
        message = row['message']
        emotes = ast.literal_eval(row['emotes'])

        for e in emotes:
            message = message.replace(e, '')
        return message.strip()
    
    # if emote is in fromt make it flush so that training doesnt take in random white space
    def front_flush(self, m):
        return m.lstrip(string.punctuation + string.whitespace)
    
    # remove repetative spam
    def is_copy_pasta(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        pattern = re.compile(r'^(.+?)\s*(\1\s*){2,}$')
        return bool(pattern.match(text))
    
    # finalize the rows (message, emote, user, weight, labeled emotion, overriden flag)
    # write to general clean dataset for modeling
    def model_ready(self, df):

        # final column choices to clean up dataset
        choice_cols = ["message", "emotes", "username","final_weights", "final_emotion"]
        final_df =  df[choice_cols]
        final_df.col_names = ["message", "emotes", "username", "weights", "emotion_label"]
        
        # removal of emotes from the x train, and standardization of text
        final_df = final_df[~final_df['message'].apply(self.is_copy_pasta)]
        final_df = final_df[~final_df['message'].str.lower().str.contains('badges=;client')]
        final_df = final_df[~final_df['message'].str.lower().str.contains('raid')]
        final_df['message'] = final_df.apply(self.remove_emotes, axis=1)
        final_df['message'] = final_df['message'].apply(self.front_flush)
        final_df = final_df.drop_duplicates(subset='message')
        final_df = final_df.reset_index(drop=True)
        final_df.to_csv(self.final_path)

    # combine two csvs for extra data
    def wombo_combo(self):
        base = pd.read_csv(self.master_record)
        add = pd.read_csv(self.extra_records)
        larger_df = pd.concat([base, add], ignore_index=True)
        larger_df.to_csv('./data/official_dataset/master_chat_8k.csv')

    # runs all steps of manipulation and final filtering/formatting
    def run_process(self):
        df = self.data_loader()
        df = self.additional_filtering(df)
        df = self.override_labels(df)p
        self.model_ready(df)


if __name__ == "__main__":
    
    # running the pain process
    emote_analyzer = emote_analyzer()
    emote_analyzer.run_process()
    
    # in case you need to add more data make changes here
    #emote_analyzer.wombo_combo()