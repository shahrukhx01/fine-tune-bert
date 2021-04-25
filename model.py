from transformers import BertModel, BertConfig



class BERTClassifier:
    def __init__(self, num_labels=2):
        self.configuration = BertConfig()

    def get_model(self):
        """
        Initialize pretrained bert model from huggingface model hub
        """
        # initializing a model from the bert-base-uncased style configuration
        model = BertModel(self.configuration)

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.model.cuda()