from src.models.elvira import Elvira2023Small
from src.models.facu import VariationalAutoencoder
from src.trainers import TrainWithSyntheticData

if __name__ == "__main__":
    model = VariationalAutoencoder("auto_simple_vae.json")
    trainer = TrainWithSyntheticData(model)

    trainer.train(10)