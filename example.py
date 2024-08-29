from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import VariationalAutoencoder, ConvolutionalAutoencoder
from src.trainers import TrainWithSyntheticData

if __name__ == "__main__":
    # model = VariationalAutoencoder("manual_simple_vae.json")
    model = Elvira2023Full()
    trainer = TrainWithSyntheticData(model)

    trainer.train(10)