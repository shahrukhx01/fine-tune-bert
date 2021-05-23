from transformers import BertModel, BertConfig
from transformers import BertForSequenceClassification


class BERTClassifier:
    def __init__(self, num_labels=2):
        self.configuration = BertConfig()

    def get_model(self):
        """
        Initialize pretrained bert model from huggingface model hub
        """
        # initializing a model from the bert-base-uncased style configuration
        model = BertModel(self.configuration)

        model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-mini", num_labels=2)
        model.cuda()

        return model