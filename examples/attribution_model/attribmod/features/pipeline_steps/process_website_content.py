import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class ProcessWebsiteContent(TransformerMixin, BaseEstimator):
    def __init__(self, map_dict=None):
        self.map_dict = map_dict

    def fit(self, X, y=None):
        return self

    def _find_max_depth(self, list_in, map_dict):
        list_out = []
        for el in list_in:
            if el in map_dict:
                list_out.append(map_dict[el])
            else:
                if "bank" in el:
                    list_out.append(map_dict["bank"])
                if "-danke" in el:
                    list_out.append(map_dict["danke"])
                if "vitality" in el:
                    list_out.append(map_dict["vitality"])
                if "1-versicherungsnehmer" in el:
                    list_out.append(map_dict["1-versicherungsnehmer"])
                if "12-berechnungsergebnis" in el:
                    list_out.append(map_dict["12-berechnungsergebnis"])
                if "5-situation" in el:
                    list_out.append(map_dict["5-situation"])
                else:
                    list_out.append(0)
        return max(list_out)

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        X["web_seiten_content"] = X["web_seiten_content"].fillna("other")

        X["web_seiten_content"] = X["web_seiten_content"].str.split(",")
        X["content_depth"] = X["web_seiten_content"].apply(
            self._find_max_depth, args=(self.map_dict,)
        )
        return X
