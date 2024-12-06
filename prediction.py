import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/kaggle/input/dataset-coursera-csv/Dataset Cousera_CSV.csv', encoding="latin-1")

def clean_course_name(text):
    if isinstance(text, str):
        text = text.strip()
        text = ' '.join(text.split())
        return text
    return None

df['Course_Name'] = df['Course_Name'].apply(clean_course_name)
df = df[df['Course_Name'].notna()]

label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['Course_Name'])

class RobertaBiLSTM_CNN_Model(nn.Module):
    def __init__(self, num_labels, hidden_size=768, lstm_hidden_size=128, lstm_layers=1, cnn_kernel_size=3, dropout_prob=0.3):
        super(RobertaBiLSTM_CNN_Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, 
                              num_layers=lstm_layers, batch_first=True, bidirectional=True)
    
        self.dropout = nn.Dropout(dropout_prob)

        # Lớp CNN
        self.cnn = nn.Conv1d(in_channels=lstm_hidden_size * 2, out_channels=64, kernel_size=cnn_kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Tính toán cho classifier
        self.classifier = nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Nhận embedding từ RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        
        # Truyền qua BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)  # (batch_size, sequence_length, lstm_hidden_size*2)

        # Dropout sau BiLSTM
        lstm_output = self.dropout(lstm_output)
        
        # Đổi kích thước cho CNN
        lstm_output = lstm_output.permute(0, 2, 1)  # (batch_size, lstm_hidden_size*2, sequence_length)
        
        # Truyền qua CNN
        cnn_output = self.cnn(lstm_output)  # (batch_size, 64, mới_dài_sequence)
        pooled_output = self.pool(cnn_output)  # (batch_size, 64, mới_dài_sequence/2)

        # Sử dụng Global Max Pooling
        pooled_output = torch.mean(pooled_output, dim=2)  # (batch_size, 64)
        
        # Nhận logits từ classifier
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        # Tính toán loss nếu labels được cung cấp
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


# Tải lại mô hình đã lưu
model = RobertaBiLSTM_CNN_Model(num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load('/kaggle/input/best-model/best_model.pth', map_location=device))
model.to(device)
model.eval()

# Tải tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def predict(texts):
    # Tokenize văn bản
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # Chuyển tokenized input sang thiết bị của mô hình
    encodings = {key: val.to(device) for key, val in encodings.items()}
    
    # Dự đoán logits từ mô hình
    outputs = model(**encodings)
    logits = outputs.logits
    
    # Chuyển logits thành nhãn dự đoán
    preds = logits.argmax(-1).cpu().numpy()
    
    # Giải mã nhãn từ số thành tên khóa học
    return [label_encoder.inverse_transform([pred])[0] for pred in preds]

# Dự đoán trên văn bản mẫu
new_texts = ["""
Hey there. By this point, you've created a few iterations of your paper wireframes. You've even explored responsive designs in your wireframes by sketching for different screen sizes. After you've explored multiple ideas for wireframes on paper, and you understand which wireframe elements will provide the best user experience, it's time to bring your paper wireframe to life digitally. While drawing wireframes on paper is fast and inexpensive, things get a little trickier when we move to digital wireframes. So make sure you feel good about your paper wireframes before you move on to the next step. To determine if you're ready to transition your paper wireframes to digital, ask yourself a few questions. One, do you have an idea of the layout you're aiming for? Two, have you received feedback from peers or managers on your paper wireframes? And three, are you ready to consider basic visual cues like size, text, and typography, or other visual elements like content or images? If you answered yes to all three of these questions, then you're ready to start creating a digital version of your paper wireframes. But here's one tip: Be sure to hang on to those paper wireframes for your portfolio so you can showcase your full design process. In the next video, you'll learn how to create and adjust columns, grids, and gutters in Adobe XD. Just like you used dot grid paper to lay out your paper wireframes, you'll use these features to lay out your digital wireframes."""]
predictions = predict(new_texts)
print(predictions)