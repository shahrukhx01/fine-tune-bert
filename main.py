from model import BERTClassifier
from config import BertOptimConfig
from train import train_model
from data_loader import SpamDataLoader


if __name__ == '__main__':
    label_map = {'spam': 1, 'ham': 0}
    epochs = 2
    num_labels = 2
    data_loaders = SpamDataLoader('spam.csv', label_map =label_map, batch_size=8)
    model = BERTClassifier(num_labels=num_labels).get_model()
    optim_config = BertOptimConfig(model=model, train_dataloader=data_loaders.train_dataloader, epochs=epochs)
    
    train_model(model=model, 
                optimizer=optim_config.optimizer, 
                scheduler=optim_config.scheduler, 
                train_dataloader=data_loaders.train_dataloader, 
                validation_dataloader=data_loaders.validation_dataloader, 
                epochs=epochs)