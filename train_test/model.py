# flake8: noqa
import os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AdamW
)


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


class ToxicCommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    

class ToxicCommentModel:
    def __init__(self, save_dir, classes):
        self.cache_dir = save_dir
        self.classes = classes

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # if local model is found, fetch it from the cache directory
        if os.path.exists(self.cache_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(self.cache_dir)
            config = AutoConfig.from_pretrained(self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.cache_dir, config=config).to(device)

        # fetch pre-trained model from the internet
        else:
            base_model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=len(self.classes),
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True
            ).to(device)

    def train(self, train_texts, train_labels):
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)
        
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False
        )

        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False
        )

        train_dataset = ToxicCommentDataset(train_encodings, train_labels)
        val_dataset = ToxicCommentDataset(val_encodings, val_labels)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.train()

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        optim = AdamW(self.model.parameters(), lr=5e-5)

        num_train_epochs = 2
        for epoch in range(num_train_epochs):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).to(torch.float32)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs[0]
                loss.backward()
                optim.step()

        self.model.eval()

    def test(self, test_texts, test_labels):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        test_encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(device)

        test_dataset = ToxicCommentDataset(test_encodings, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        for batch in test_loader:
            with torch.no_grad():
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

                labels = torch.sigmoid(outputs.logits).squeeze(dim=0) > 0.5
                labels = labels.tolist()
                print(f"Hamming score = {hamming_score(batch['labels'], labels)}")

                predictions = F.softmax(outputs.logits, dim=1)
                labels = torch.argmax(predictions, dim=1)
                labels = [self.model.config.id2label[label_id] for label_id in labels.tolist()]

    def save(self):
        self.tokenizer.save_pretrained(self.cache_dir)
        self.model.save_pretrained(self.cache_dir)