#Import các thư viện cần thiết
import pandas as pd
import re
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.modeling_outputs import SequenceClassifierOutput

# Load dataset
df = pd.read_csv('/kaggle/input/dataset-coursera-csv/Dataset Cousera_CSV.csv', encoding="latin-1")

df.at[348,'Course_Content']

# Clean the Course_Name and Course_Content columns
def clean_course_name(text):
    if isinstance(text, str):  
        text = text.strip()  # Xóa khoảng trắng ở đầu và cuối
        text = ' '.join(text.split())  # Xóa khoảng cách dư thừa
        return text
    else:
        return None

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\[MUSIC\]|\[SOUND\]', '', text)
        text = re.sub(r'\sen\b', '', text)
        text = re.sub(r'Play video starting at.*?follow transcript\d+:\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    else:
        return ''

# Áp dụng dữ liệu đã làm sạch vào data
df['Course_Name'] = df['Course_Name'].apply(clean_course_name)
df['Course_Content'] = df['Course_Content'].apply(clean_text)

# Xóa các giá trị bị thiếu ở cột CourseName
df = df[df['Course_Name'].notna()]

df.at[348,'Course_Content']

# Encode labels (Course_Name)
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['Course_Name'])

# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Course_Content'].tolist(), df['labels'].tolist(), test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize text data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert labels to torch tensors
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Tạo lớp tập dữ liệu
class CourseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Tạo tập dữ liệu huấn luyện và kiểm tra
train_dataset = CourseDataset(train_encodings, train_labels)
test_dataset = CourseDataset(test_encodings, test_labels)

class RobertaBiLSTM_CNN_Model(nn.Module):
    def __init__(self, num_labels, hidden_size=768, lstm_hidden_size=256, lstm_layers=2, cnn_kernel_size=3):
        super(RobertaBiLSTM_CNN_Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, 
                              num_layers=lstm_layers, batch_first=True, bidirectional=True)
        
        # Thêm lớp CNN
        self.cnn = nn.Conv1d(in_channels=lstm_hidden_size * 2, out_channels=64, kernel_size=cnn_kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Tự động tính toán kích thước đầu vào cho lớp Linear
        self.classifier = nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Nhận embedding từ RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        
        # Truyền embedding qua BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)  # (batch_size, sequence_length, lstm_hidden_size*2)

        # Đổi kích thước cho lớp CNN
        lstm_output = lstm_output.permute(0, 2, 1)  # (batch_size, lstm_hidden_size*2, sequence_length)
        
        # Truyền qua lớp CNN
        cnn_output = self.cnn(lstm_output)  # (batch_size, 64, mới_dài_sequence)
        pooled_output = self.pool(cnn_output)  # (batch_size, 64, mới_dài_sequence/2)

        # Sử dụng Global Max Pooling để lấy đầu ra có kích thước cố định
        pooled_output = torch.mean(pooled_output, dim=2)  # (batch_size, 64)
        
        # Nhận logits từ bộ phân loại
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        # Tính toán loss nếu labels được cung cấp (giai đoạn huấn luyện)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    
    # Kiểm tra thiết bị có sẵn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sử dụng mô hình mới
model = RobertaBiLSTM_CNN_Model(num_labels=len(label_encoder.classes_))
# Chuyển mô hình sang thiết bị đã xác định
model.to(device)

# Xác định các hàm tính toán
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#Các đối số huấn luyện
from transformers import EarlyStoppingCallback
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the model
trainer.train()

# Đánh giá mô hình
results = trainer.evaluate()
print(results)

# Lưu mô hình tùy chỉnh
torch.save(model.state_dict(), 'best_model.pth')
# Lưu bộ tokenizer
tokenizer.save_pretrained('./best_model')

# Hàm dự đoán trên văn bản tùy chỉnh
def predict(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    encodings = {key: val.to(device) for key, val in encodings.items()}
    outputs = model(**encodings)
    logits = outputs.logits
    preds = logits.argmax(-1).cpu().numpy()
    return [label_encoder.inverse_transform([pred])[0] for pred in preds]

#Predict
new_texts = [""" 
Refresher time. Remember that DNS is a global system managed in a tiered hierarchy with ICANN at the top level. Domain names need to be globally unique for a global system like this to work. You can't just have anyone decide to use any domain name, it'd be chaos. Enter the idea of a registrar, an organization responsible for assigning individual domain names to other organizations or individuals. Originally, there were only a few registrars. The most notable was a company named Network Solutions Inc. It was responsible for the registration of almost all domains that weren't country-specific. As the popularity of the internet grew, there was eventually enough market demand for competition in this space. Finally, the United States government and Network Solutions, Inc came to an agreement to let other companies also sell domain names. Today, there are hundreds of companies like this all over the world. Registering a domain name for use is pretty simple. Basically, you create an account with the registrar, use their web UI to search for a domain name to determine if it's still available, then you agree upon a price to pay and the length of your registration. Once you own the domain name, you can either have the registrar's name servers act as the authoritative name servers for the domain, or you can configure your own servers to be authoritative. Domain names can also be transferred by one party to another and from one registrar to another. The way this usually works is that the recipient registrar will generate a unique string of characters to prove that you own the domain and that you're allowed to transfer it to someone else. You'd configure your DNS settings to contain this string in a specific record, usually a text record. Once this information has propagated, it can be confirmed that you both own the domain and approve its transfer. After that, ownership would move to the new owner or registrar. An important part of the domain name registration is that these registrations only exist for a fixed amount of time. You typically pay to register domain names for a certain number of years. It's important to keep on top of when your domain names might expire because once they do, they're up for grabs and anyone else could register them.

"""] 
predictions = predict(new_texts)
print(predictions)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict on the test dataset
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
true_labels = test_labels.numpy()

# Compute confusion matrix
cm = confusion_matrix(true_labels, preds)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))

# Plot heatmap with color bar
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, 
            annot_kws={"size": 20})  # Adjust font size for numbers in the heatmap

# Get the color bar and adjust its font size
cbar = plt.gcf().axes[-1]  # Access the color bar (last axis in the figure)
cbar.tick_params(labelsize=20)  # Adjust the font size of the color bar numbers

# Set font size for x and y tick labels
plt.xticks(fontsize=18)  # Adjust font size of x-axis tick labels
plt.yticks(fontsize=18)  # Adjust font size of y-axis tick labels

# Set labels and title
plt.xlabel('Predicted Labels', fontsize=18)
plt.ylabel('True Labels', fontsize=18)
plt.title('Confusion Matrix', fontsize=25)

# Save confusion matrix to a file (optional)
# plt.savefig('z_confusion_matrix_RoBERTa_BiLSTM_CNN_k3.png', dpi=300, bbox_inches='tight')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict on the test dataset
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
true_labels = test_labels.numpy()

# Compute confusion matrix
cm = confusion_matrix(true_labels, preds)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot Normalized Confusion Matrix
plt.figure(figsize=(10, 7))

# Plot heatmap with color bar
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, 
            annot_kws={"size": 20})  # Adjust font size for numbers in the heatmap

# Get the color bar and adjust its font size
cbar = plt.gcf().axes[-1]  # Access the color bar (last axis in the figure)
cbar.tick_params(labelsize=18)  # Adjust the font size of the color bar numbers

# Set font size for x and y tick labels
plt.xticks(fontsize=18)  # Adjust font size of x-axis tick labels
plt.yticks(fontsize=18)  # Adjust font size of y-axis tick labels

# Set labels and title
plt.xlabel('Predicted Labels', fontsize=18)
plt.ylabel('True Labels', fontsize=18)
plt.title('Normalized Confusion Matrix', fontsize=20)

# Save confusion matrix to a file (optional)
# plt.savefig('z_Normalized_confusion_matrix_RoBERTa_BiLSTM_CNN_k3.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()