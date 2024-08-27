from src.models.elvira import Elvira2023Small
from src.trainers import TrainWithDataloader

if __name__ == "__main__":
    model = Elvira2023Small(block_size=16)
    trainer = TrainWithDataloader(model)

    trainer.train(5)