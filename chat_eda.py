import pandas as pd
import numpy as np
from sklearn.utils import resample

class chat_eda:

    def __init__(self):
        self.data_path = './data/official_dataset/model_ready_pure.csv'
        self.data_path_v2 = './data/official_dataset/model_ready_data_v2.csv'
        self.seed = 741

    # intial exploration to determine distributions
    def data_exploration(self):

        df = pd.read_csv(self.data_path, encoding="utf-8" , index_col=0)
        print(df)
        print(df.shape[0])
        print(df['final_emotion'].value_counts())

        unique_user_count = df['username'].nunique()
        print(f"Unique users: {unique_user_count}")

        df = pd.read_csv(self.data_path_v2, encoding="utf-8" , index_col=0)
        print(df)
        print(df.shape[0])
        print(df['final_emotion'].value_counts())

        unique_user_count = df['username'].nunique()
        print(f"Unique users: {unique_user_count}")

    # fixes distributions issues
    # samples all of v2 and randomly samples hapiness and surprise to match the 3rd tier
    def level_out(self):

        # combine both datasets
        base = pd.read_csv(self.data_path)
        df = base
        #add = pd.read_csv(self.data_path_v2)
        #df = pd.concat([base, add], ignore_index=True)
        #df.to_csv('./official_dataset/model_ready_pure.csv')
        #print(df)
        #print(df.shape[0])
        #print(df['final_emotion'].value_counts())

        # find the limiting amount and limit classes above to that number by random sample
        # with current samples lets try and down/up sample to ~1500 per emotion
        
        # downsample to 4000
        df_hapiness = df[df['final_emotion'] == 'hapiness'].sample(n=2504, random_state=self.seed)
        #df_surprise = df[df['final_emotion'] == 'surprise'].sample(n=1500, random_state=self.seed)

        # keep neutral due to being close to 1500
        df_neutral = df[df['final_emotion'] == 'neutral']
        
        # over sampling the minority to 2x its original quantity
        #df_sadness = df[df['final_emotion'] == 'sadness']
        #df_disgust = df[df['final_emotion'] == 'disgust']
        #df_fear = df[df['final_emotion'] == 'fear']
        #df_anger = df[df['final_emotion'] == 'anger']

        #df_sadness = resample(df_sadness, replace=True, n_samples=round(int(df_sadness.shape[0]) * 1.7), random_state=self.seed)
        #df_disgust = resample(df_disgust, replace=True, n_samples=int(df_disgust.shape[0]) * 2, random_state=self.seed)
        #df_fear = resample(df_fear, replace=True, n_samples=int(df_fear.shape[0]) * 2, random_state=self.seed)
        #df_anger = resample(df_anger, replace=True, n_samples=int(df_anger.shape[0]) * 2, random_state=self.seed)

        # no oversampling this time just querying the top 4 emotions
        df_sadness = df[df['final_emotion'] == 'sadness']
        df_surprise = df[df['final_emotion'] == 'surprise']

        df_balanced = pd.concat([
            df_hapiness,
            df_surprise,
            df_neutral,
            df_sadness
            #df_disgust,
            #df_fear,
            #df_anger
        ])
        print(df_balanced.shape[0])
        print(df_balanced['final_emotion'].value_counts())
        
        df_balanced.to_csv('./official_dataset/even_more_balance_2.csv')
        # shuffle the dataset

        '''
        data imbalance strategy
        Happiness | 9490 | Downsample | 1500
        Surprise | 2504 | Downsample | 1500
        Neutral | 1482 | Keep | 1482
        Sadness | 854 | SMOTE ×2 | 1700
        Disgust | 549 | SMOTE ×2 | 1100
        Fear | 196 | SMOTE ×2 | 392
        Anger | 183 | SMOTE ×2 | 366
        '''

    def data_details(self):

        import matplotlib.pyplot as plt
        import seaborn as sns
        #df = pd.read_csv('./official_dataset/even_more_balance_2.csv')
        #print(df.shape[0])
        df = pd.read_csv('./official_dataset/model_ready_pure.csv')
        sns.countplot(x='final_emotion', data=df)
        plt.title("Final Emotion Distribution")
        plt.show()

        from collections import Counter
        from ast import literal_eval

        df['emotes'] = df['emotes'].apply(literal_eval)
        emotes_by_class = df.groupby('final_emotion')['emotes'].sum()
        for emotion, emotes in emotes_by_class.items():
            top = Counter(emotes).most_common(7)
            print(f"{emotion}: {top}")
        # interesting stats and images
        # average word counts
        # number of unique users
        # distribution graph of assigned emotions
        # total distribution of weights
        # most common emotes top 10
        # outlier distribution of sorts
        # describe
        import ast
        emotions = ['hapiness', 'neutral', 'sadness', 'surprise']
        for e in emotions:
            df[e] = df['final_weights'].apply(lambda x: ast.literal_eval(x).get(e, 0))


        unique_user_count = df['username'].nunique()
        print(f"Unique users: {unique_user_count}")

        # Now plot heatmap
        mean_scores = df.groupby('final_emotion')[emotions].mean()
        sns.heatmap(mean_scores, annot=True, cmap="Blues")
        plt.title("Average Emotion Weight per Final Emotion")
        plt.show()
        '''
        from sklearn.manifold import TSNE

        X = df[emotions].values
        X_embedded = TSNE(n_components=2).fit_transform(X)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=df['final_emotion'])
        plt.title("t-SNE on Emotion Vectors")
        '''
        from sklearn.ensemble import IsolationForest
        from sklearn.manifold import TSNE

        X = df[emotions].values
        X_emb = TSNE(n_components=2, random_state=42).fit_transform(X)

        iso = IsolationForest(contamination=0.05)
        df['outlier'] = iso.fit_predict(X)

        sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=df['outlier'], palette={1: 'lightblue', -1: 'darkblue'})
        plt.title("t-SNE with Outlier Detection")
        plt.show()

        df['msg_length'] = df['message'].apply(lambda x: len(str(x).split()))
        sns.boxplot(x='final_emotion', y='msg_length', data=df)
        plt.title("Message Length by Emotion")
        plt.show()

        #code for highest chatting user
        
        # average vocabulary richness (unique word ratio)


if __name__ == "__main__":
    
    chat_eda = chat_eda()
    #chat_eda.data_exploration()
    #chat_eda.level_out()
    #df = pd.read_csv("./official_dataset/even_more_balance_2.csv")
    #df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    #print(df_shuffled[df_shuffled['final_emotion'] == "hapiness"].head(10))
    #print(df_shuffled[df_shuffled['final_emotion'] == "sadness"].head(5))
    #print(df_shuffled[df_shuffled['final_emotion'] == "surprise"].head(5))
    #print(df_shuffled[df_shuffled['final_emotion'] == "neutral"].head(10))
    chat_eda.data_details()