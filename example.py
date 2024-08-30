import torch
from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import VariationalAutoencoder, ConvolutionalAutoencoder
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def train_with_synth_data():
    # model = VariationalAutoencoder("manual_simple_ae.json")
    model = ConvolutionalAutoencoder("manual_simple_ae.json")
    # model.load_state_dict(torch.load("vae_synth_data.pth"))
    trainer = TrainWithSyntheticData(model)
    try:
        trainer.train(1_000_000, 5e-4)
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), "cae_synth_data.pth")


def finetune_with_real_data():
    # model = VariationalAutoencoder("manual_simple_ae.json")
    model = ConvolutionalAutoencoder("manual_simple_ae.json")
    model.load_state_dict(torch.load("cae_synth_data.pth"))
    trainer = TrainWithRealData(model)
    try:
        trainer.train(100_000, 1e-4)
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), "cae_real_data_finetuned.pth")


if __name__ == "__main__":
    train_with_synth_data()