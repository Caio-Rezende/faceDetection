from models_loader import loader
from process.args import get_args
from process.models import clear

args = get_args()


def call():
    repeat = True

    while repeat:
        repeat = False
        models = loader.load()
        for model in models:
            filtered = list(filter(lambda a: a.name ==
                                   model.name, models))

            if len(filtered) <= args.threshold:
                clear(model.name, filtered, models)

                repeat = True
                del filtered
                break

        del models
    del repeat
