from src.model.gpt_language_model.gpt import GPTLanguageModel
from aihwkit.nn.conversion import convert_to_analog
from config.nanogpt_config import tiny_cfg
from config.rpu_config import rpu_config

def build_model():
    tiny = GPTLanguageModel(**tiny_cfg)
    print(tiny)
    return tiny

def convert_digital_to_analog(model):
    analog = convert_to_analog(model, rpu_config)
    print(analog)
    return analog


