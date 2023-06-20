from models_loader import loader
from process.args import get_args
from process.models import models_fix

args = get_args()


def call():
    totalIndices = len(args.indices)
    totalUnknowns = len(args.unknowns)
    totalNames = len(args.names)

    if totalIndices > 0 and totalUnknowns > 0:
        print('Only invoke with indices or unknowns, not both !!!')
        del totalUnknowns, totalIndices, totalNames
        return

    if totalIndices > 0 and (totalNames == 1 or totalIndices == totalNames):
        del totalUnknowns

        names = get_names(totalIndices)

        del totalIndices, totalNames

        [models_fix(index, name)
         for index, name in zip(args.indices, names)]
        
        del names

        return

    if totalUnknowns > 0 and (totalNames == 1 or totalUnknowns == totalNames):
        del totalIndices

        names = get_names(totalUnknowns)

        models = loader.load()

        for unknownIndex, unknown in enumerate(args.unknowns):
            for index, model in enumerate(models):
                if model.name == f'unknown-{unknown:05}':
                    models_fix(index, names[unknownIndex])

        del totalUnknowns, totalNames, models, names

        return


def get_names(compareTo: int):
    totalNames = len(args.names)

    names = args.names
    if totalNames == 1 and totalNames != compareTo:
        name = f'{names[0]}'
        names = [name for x in range(compareTo)]
        del name

    del totalNames

    return names
