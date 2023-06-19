import process_models
from process_parser import get_args

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

        [process_models.models_fix(index, name)
         for index, name in zip(args.indices, names)]

    if totalUnknowns > 0 and (totalNames == 1 or totalUnknowns == totalNames):
        del totalIndices

        names = get_names(totalUnknowns)

        models = process_models.load_models()

        for unknownIndex, unknown in enumerate(args.unkowns):
            for index, model in enumerate(models):
                if model.name == f'unknown-{unknown:03}':
                    process_models.models_fix(index, names[unknownIndex])

        del totalUnknowns, totalNames, models


def get_names(compareTo: int):
    totalNames = len(args.names)

    names = args.names
    if totalNames == 1 and totalNames != compareTo:
        name = f'{names[0]}'
        names = [name for x in range(compareTo)]
        del name

    del totalNames

    return names
