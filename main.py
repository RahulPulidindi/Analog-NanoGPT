
from model import build_model, convert_digital_to_analog
from train import train
import wandb

def main():
    digital_model = build_model()
    print("Starting Digital training")
    model = train("Digital_NanoGPT", digital_model, num_epochs=100, batch_size=16)
    wandb.finish()

    print("Starting analog model training")
    analog_model = convert_digital_to_analog(model)
    train("Analog_NanoGPT", analog_model, num_epochs=5)
    wandb.finish()

main()
