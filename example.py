import torch
from src.models.elvira import Elvira2023Small, Elvira2023Full
from src.models.facu import VariationalAutoencoder, ConvolutionalAutoencoder
from src.trainers import TrainWithSyntheticData, TrainWithRealData

def get_class_name(instance) -> str:
    return repr(instance.__class__).strip("'>").split(".")[-1]

def train_with_synth_data():
    # model = VariationalAutoencoder("manual_simple_ae.json")
    model = VariationalAutoencoder("manual_simple_ae.json")
    # model.load_state_dict(torch.load("vae_synth_data.pth"))
    trainer = TrainWithSyntheticData(model)
    try:
        trainer.train(r"C:\Users\fq\Facu\smoe\src\trainers\configs\simple_training.json")
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), f"{get_class_name(model)}_synth_data.pth")


def finetune_with_real_data():
    # model = VariationalAutoencoder("manual_simple_ae.json")
    model = VariationalAutoencoder("manual_simple_ae.json")
    model.load_state_dict(torch.load(f"{get_class_name(model)}_synth_data.pth"))
    trainer = TrainWithRealData(model)
    try:
        trainer.train(r"C:\Users\fq\Facu\smoe\src\trainers\configs\simple_training.json")
    except Exception as e:
        print(e)
    finally:
        torch.save(model.state_dict(), f"{get_class_name(model)}_finetune_real_data.pth")


if __name__ == "__main__":
    # train_with_synth_data()
    finetune_with_real_data()