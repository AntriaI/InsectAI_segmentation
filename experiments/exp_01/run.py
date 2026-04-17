from src.datafeeder import get_dataloader
from src.train import train
from src.models import BinaryClassifier
from src.utils import Configuration


if __name__=="__main__":
    config_path = "experiments/exp_01/config.json"
    config = Configuration(config_path) 

    num_classes = 1
    model = BinaryClassifier(num_classes)

    
    train_loader, test_loader = get_dataloader(config)



    train(config=config, model=model, 
          train_loader=train_loader, 
          val_loader=test_loader)