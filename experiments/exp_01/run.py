from src.datafeeder import get_dataloader
from src.train import train
from src.models import UNetResNet34
from src.utils import Configuration


if __name__ == "__main__":
    config_path = "experiments/exp_01/config.json"
    config = Configuration(config_path)

    model = UNetResNet34(num_classes=1, pretrained=True)

    train_loader, val_loader = get_dataloader(config)

    train(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader)