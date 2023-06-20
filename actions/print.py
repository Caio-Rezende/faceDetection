from models_loader import loader
from process.args import get_args

args = get_args()


def call():
    models = loader.load()
    for model in models:
        print(model)


def verbose(content: str):
    if args.verbose:
        print(content)
