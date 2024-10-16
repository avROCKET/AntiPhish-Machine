import torch
import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from tqdm.auto import tqdm
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

if torch.cuda.is_available():
    print("GPU is available.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")  
else:
    print("GPU is not available, using CPU instead.")

# using wordnet from NLTK to replace synonyms, to create a variety of data using my existing data.
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalnum() or char == ' '])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalpha()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  
            break

    sentence = ' '.join(new_words)
    return sentence

# loading dataset
df = pd.read_csv('.\\transformer_model_assests\\phishing_data_by_type_updated - transformer.csv', encoding='ISO-8859-1', usecols=['Text', 'Type'])

# convert label to number
df['label'] = df['Type'].map({'Safe': 0, 'Phishing': 1})

print(df.head())

# this applies the data augmentation (synonym replacement) to the dataset
augmented_texts = df['Text'].apply(lambda x: synonym_replacement(x, n=2))
df_augmented = df.copy()
df_augmented['Text'] = augmented_texts
df_combined = pd.concat([df, df_augmented])

# training the combined dataframe
train_df, val_df = train_test_split(df_combined, test_size=0.1, stratify=df_combined['label'])


train_texts, train_labels = train_df['Text'].tolist(), train_df['label'].tolist()
val_texts, val_labels = val_df['Text'].tolist(), val_df['label'].tolist()



tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = EmailDataset(train_texts, train_labels, tokenizer)
val_dataset = EmailDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# calculate accuracy
def calculate_accuracy(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return accuracy_score(labels, np.argmax(preds, axis=1))

# training and validation
def train_and_validate(model, train_loader, val_loader, optimizer, device, epochs=5): # 5 epochs for current model
    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0 # loss and accuracy

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            logits = outputs.logits
            total_accuracy += calculate_accuracy(logits, batch['labels'])

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_accuracy / len(train_loader)

        # validation loss and accuracy
        model.eval()
        val_loss, val_accuracy = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                val_accuracy += calculate_accuracy(logits, batch['labels'])

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_accuracy / len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

train_and_validate(model, train_loader, val_loader, optimizer, device) # begin training

model_path = "C:\\Users\\justi\\OneDrive\\Assignments of Fall 2019\\Capstone Project" # to save/load, in this case just load.

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Test email text, will replace to variable, 'email_content'
email_text = """
Per the User Agreement, Section 9, we may immediately issue a warning,
temporarily suspend, indefinitely suspend or terminate your membership
and refuse to provide our services to you if we believe that your
actions may cause financial loss or legal liability for you, our users
or us. We may also take these actions if we are unable to verify or
authenticate any information you provide to us.


Thank you for your prompt attention to this matter. Please understand that this is a security measure meant to help protect you and your account.
<br><br>
Regards,
<br>
Safeharbor Department<br>
Visa Card, Inc.<br><br>
"""
# tokenize and prepare inputs
inputs = tokenizer(email_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

inputs = {key: value.to(model.device) for key, value in inputs.items()}

# prediction
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

probabilities = torch.softmax(logits, dim=-1)

prob_safe, prob_phishing = probabilities[0].tolist()

print(f"Probability of being safe: {prob_safe:.4f}")
print(f"Probability of being phishing: {prob_phishing:.4f}")

