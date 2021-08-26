from transformers import BertForSequenceClassification, BertTokenizer


model = BertForSequenceClassification.from_pretrained("./model")
model.push_to_hub("buy-sell-intent-classifier-bert-mini")

tokenizer = BertTokenizer.from_pretrained("./model")
tokenizer.push_to_hub("buy-sell-intent-classifier-bert-mini")
