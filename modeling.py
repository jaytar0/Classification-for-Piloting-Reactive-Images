import pandas as pd
import numpy as np
from dotenv import load_dotenv

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split
from distil_data import distil_data
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import fasttext
load_dotenv()

class emote_model:

    def __init__(self):
        self.data_path = './data/official_dataset/even_more_balance_2.csv'
        self.label_encoder = LabelEncoder()
        self.seed = 741
        
    def data_loader(self, hold_out=0.2):
        
        df = pd.read_csv(self.data_path)
        X, y = df["message"], df["final_emotion"]
        y_encode = self.label_encoder.fit_transform(y)

        # 80/20 split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encode, test_size=hold_out, 
                                                            stratify=y_encode, random_state=self.seed)
        
        self.X_train = self.X_train.astype(str)
        self.X_test = self.X_test.astype(str)
        self.y_train = self.y_train.astype(str)
        self.y_test = self.y_test.astype(str)


    # TF-IDF (LR, RF, XG, SVM)
    def termfreq_lr(self):

        lr_pipe = Pipeline([('tfidf', TfidfVectorizer()), 
                        ('clf', LogisticRegression(multi_class='multinomial', 
                                                   solver='lbfgs', max_iter=1000, class_weight='balanced'))])
        lr_params = {
            'tfidf__ngram_range': [(1,1)],
            'tfidf__max_features': [10000, 20000],
            'tfidf__min_df': [1, 3],
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['lbfgs']
        }

        grid_search_lr = GridSearchCV(lr_pipe, lr_params, cv=5, scoring='accuracy', n_jobs=-1, error_score = 'raise')
        grid_search_lr.fit(self.X_train, self.y_train)
        
        y_pred = grid_search_lr.predict(self.X_test)
        print("Logistic Regression Best Params:", grid_search_lr.best_params_)
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_lr.tex', 'w') as f:
            f.write(cm)
        return classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)

    
    def termfreq_rf(self):

        rf_pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(class_weight='balanced'))
        ])

        rf_params = {
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 20],
            'clf__min_samples_split': [2, 5]
        }

        grid_search_rf = GridSearchCV(rf_pipe, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_rf.fit(self.X_train, self.y_train)

        y_pred = grid_search_rf.predict(self.X_test)
        print("Random Forest Best Params:", grid_search_rf.best_params_)
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_rf.tex', 'w') as f:
            f.write(cm)

        return classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)

    def termfreq_xg(self):
        xg_pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', XGBClassifier(
                objective='multi:softprob',
                num_class=7,
                eval_metric='mlogloss'
            ))
        ])

        xg_params = {
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.05, 0.1]
        }

        y_int_train = self.y_train.astype(int)
        y_int_test = self.y_test.astype(int)
        xgb_grid = GridSearchCV(xg_pipe, xg_params, cv=5, scoring='accuracy')
        xgb_grid.fit(self.X_train, y_int_train)

        y_pred = xgb_grid.predict(self.X_test)
        print("XGBoost Best Params:", xgb_grid.best_params_)
        print(classification_report(y_int_test, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(y_int_test, y_pred)
        print(cm)
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_xg.tex', 'w') as f:
            f.write(cm)

        return classification_report(y_int_test, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)

    def termfreq_svm(self):
        svm_pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC())
        ])

        svm_params = {
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto']
        }

        svm_grid = GridSearchCV(svm_pipe, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(self.X_train, self.y_train)

        y_pred = svm_grid.predict(self.X_test)
        print("SVM Best Params:", svm_grid.best_params_)
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_svm.tex', 'w') as f:
            f.write(cm)

        return classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)
   
    # Naieve Bayes + Bag of Words Baseline
    def nb_bow(self):

        nb_pipe = Pipeline([
            ('bow', CountVectorizer()),  # Bag of Words
            ('nb', MultinomialNB())      # Naive Bayes
        ])

        nb_params = {
            'bow__ngram_range': [(1,1), (1,2)],
            'bow__min_df': [1, 3],
            'nb__alpha': [0.5, 1.0, 1.5]
        }

        grid_search_nb = GridSearchCV(nb_pipe, nb_params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_nb.fit(self.X_train, self.y_train)


        y_pred = grid_search_nb.predict(self.X_test)
        print("Naive Bayes Best Params:", grid_search_nb.best_params_)
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_nb.tex', 'w') as f:
            f.write(cm)
        return classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)
    
    # distilBERT Finetuning (state of the art)
    def dbert_enhance(self):
        
        # loading data
        distil_df = pd.read_csv(self.data_path)
        distil_df = distil_df[['message','final_emotion']].dropna()
        distil_df['label'] = self.label_encoder.fit_transform(distil_df['final_emotion'])
        
        # mapping
        label_2_emotion = dict(enumerate(self.label_encoder.classes_))
        emotion_2_label = {x: y for y , x in label_2_emotion.items()}

        d_token = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        encodings = d_token(list(distil_df['message']), truncation=True, padding=True, max_length=128)

        temp_data = distil_data(encodings, distil_df['label'].to_list())
        train_size = int(0.8 * len(temp_data))
        val_size = len(temp_data) - train_size
        train_data, val_data = random_split(temp_data, [train_size, val_size])

        #ref_mod = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)
        base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

        # tune these for epochs, learning rates, batch size, weight decay
        
        args = TrainingArguments(
            output_dir='./results-lora',
            #output_dir='./results',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_dir='./logs',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=6,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_steps=100,
            learning_rate=5e-5,
            warmup_steps=100
        )
        
        lora_args = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_lin", "v_lin", "k_lin"],
            lora_dropout=0.15,
            bias="none",
            task_type=TaskType.SEQ_CLS
         )
        
        lora_mod = get_peft_model(base_model, lora_args)
        for param in base_model.parameters():
            param.requires_grad = False

        # Ensure that LoRA layers require gradients
        for param in lora_mod.parameters():
            param.requires_grad = True
        lora_mod.print_trainable_parameters()
        
        trainer = Trainer(
            model=lora_mod,
            #model=ref_mod,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=d_token,
            compute_metrics=self.bert_report
        )

        trainer.train()
        trainer.evaluate()

        lora_mod.save_pretrained("distilbert-emote-lora")
        d_token.save_pretrained("distilbert-emote-lora")       
        #ref_mod.save_pretrained("distilbert-emote-refine-model")
        #d_token.save_pretrained("distilbert-emote-refine-token")
        preds_output = trainer.predict(val_data)

        # Predictions
        y_pred = preds_output.predictions.argmax(axis=1)
        y_true = preds_output.label_ids
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_lora.tex', 'w') as f:
            f.write(cm)
        history = trainer.state.log_history

        train_losses = []
        eval_losses = []
        eval_f1s = []
        epochs = []

        for record in history:
            if 'loss' in record:
                train_losses.append(record['loss'])
            if 'eval_loss' in record:
                eval_losses.append(record['eval_loss'])
            if 'eval_f1' in record:
                eval_f1s.append(record['eval_f1'])
            if 'epoch' in record:
                epochs.append(record['epoch'])

        plt.figure(figsize=(10,6))
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(eval_losses)), eval_losses, label='Validation Loss')
        plt.xlabel('Logging Step')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()          
        return classification_report(y_true, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)

    def bert_report(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro')
        }

    # fasttext
    def train_fasttext(self, test_size=0.2, epochs=25, learning_rate=0.1, ngrams=2):

        distil_df = pd.read_csv(self.data_path)
        distil_df = distil_df[['message','final_emotion']].dropna()
        distil_df['label'] = self.label_encoder.fit_transform(distil_df['final_emotion'])
        label_2_emotion = dict(enumerate(self.label_encoder.classes_))
        emotion_2_label = {x: y for y , x in label_2_emotion.items()}

        train_df, val_df = train_test_split(distil_df, test_size=test_size, random_state=self.seed)
        
        def save_fasttext_format(df, file_path):
            with open(file_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"__label__{label_2_emotion[row['label']]} {row['message']}\n")
        
        ft_path = './fasttext/'
        save_fasttext_format(train_df, ft_path + 'train.txt')
        save_fasttext_format(val_df, ft_path + 'valid.txt')

        model = fasttext.train_supervised(input=ft_path + "train.txt", epoch=epochs, lr=learning_rate, wordNgrams=ngrams, loss='ns')

        y_true = val_df['label'].tolist()
        y_pred = [model.predict(msg)[0][0].replace('__label__', '') for msg in val_df['message']]
        y_pred = [emotion_2_label[label] for label in y_pred]
        
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        print(f"Validation F1 Score: {f1_macro}")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        cm = pd.DataFrame(cm,index = self.label_encoder.classes_, columns= self.label_encoder.classes_)
        cm = cm.to_latex(index=False)
        with open('confusion_matrix_fasttext.tex', 'w') as f:
            f.write(cm)

        return classification_report(y_true, y_pred, target_names=self.label_encoder.classes_ , output_dict=True)


if __name__ == "__main__":
    
    emote_model = emote_model()
    emote_model.data_loader()

    model_results = []
    model_name = ["TF-IDF: Logistic Regression", "TF-IDF: Random Forest", "TF-IDF: XG-Boost",
                   "TD-IDF: SVM", "TD-IDF: Naieve + BoW", "DistilBERT-LoRa", "Fast Text"]
    
    model_results.append(emote_model.termfreq_lr())
    model_results.append(emote_model.termfreq_rf())
    model_results.append(emote_model.termfreq_xg())
    model_results.append(emote_model.termfreq_svm())
    model_results.append(emote_model.nb_bow())
    model_results.append(emote_model.dbert_enhance())
    model_results.append(emote_model.train_fasttext())

    f1_table = pd.DataFrame()
    precision_table = pd.DataFrame()
    recall_table = pd.DataFrame()
    support_table = pd.DataFrame()
    
    for model, report in zip(model_name, model_results):
        # Grab just the F1-scores
        f1_scores = {
            label: metrics['f1-score'] 
            for label, metrics in report.items() 
            if isinstance(metrics, dict) and 'f1-score' in metrics
        }
        
        precision_scores = {
            label: metrics['precision']
            for label, metrics in report.items()
            if isinstance(metrics, dict) and 'precision' in metrics
        }
        recall_scores = {
            label: metrics['recall']
            for label, metrics in report.items()
            if isinstance(metrics, dict) and 'recall' in metrics
        }
        support_counts = {
            label: metrics['support']
            for label, metrics in report.items()
            if isinstance(metrics, dict) and 'support' in metrics
        }

        f1_table[model] = pd.Series(f1_scores).round(3)
        precision_table[model] = pd.Series(precision_scores).round(3)
        recall_table[model] = pd.Series(recall_scores).round(3)
        support_table[model] = pd.Series(support_counts).astype(int)

    desired_order = [label for label, metrics in model_results[0].items() 
                 if isinstance(metrics, dict) and 'f1-score' in metrics]
    
    f1_table = f1_table.reindex(desired_order).transpose()
    precision_table = precision_table.reindex(desired_order).transpose()
    recall_table = recall_table.reindex(desired_order).transpose()
    support_table = support_table.reindex(desired_order).transpose()

    # Display/store
    print(f1_table)
    print(precision_table)
    print(recall_table)
    print(support_table)

    # Output transposed tables to LaTeX format
    f1_latex = f1_table.to_latex(index=True, caption="F1 Across Models", float_format="%.3f")
    precision_latex = precision_table.to_latex(index=True, caption="Precision Across Models", float_format="%.3f")
    recall_latex = recall_table.to_latex(index=True, caption="Recall Across Models", float_format="%.3f")
    support_latex = support_table.to_latex(index=True, caption="Support Across Models")

    # Optionally save the LaTeX code to a file
    latex_path = './final_results_tex/'
    
    with open(latex_path + 'f1_table.tex', 'w') as f:
        f.write(f1_latex)

    with open(latex_path + 'precision_table.tex', 'w') as f:
        f.write(precision_latex)

    with open(latex_path + 'recall_table.tex', 'w') as f:
        f.write(recall_latex)

    with open(latex_path + 'support_table.tex', 'w') as f:
        f.write(support_latex)


