import os
import pickle

from definitions import SingletonMeta, RecognitionModel, pkl_file


class ModelsLoader(metaclass=SingletonMeta):
    def __init__(self):
        self.models: list[RecognitionModel] = []

    def load(self):
        if len(self.models) == 0 and os.path.isfile(pkl_file):
            self.models = pickle.load(pkl_file.open(mode='rb'))

        return self.models

    def save(self, models: list[RecognitionModel]):
        self.models = models
        pickle.dump(models, pkl_file.open(mode='wb'))


loader = ModelsLoader()
