import torch
import pandas as pd
import numpy as np
import nltk
import pymorphy2
from tqdm import trange
from importlib.resources import path
from transformers import AutoTokenizer, AutoModel
from numba import jit
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()


@jit(nopython=True, cache=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray) -> np.float64:
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 1.
    return np.dot(u, v)


class Digest:

    def __init__(self, path_labse: path):
        self.model_labse = AutoModel.from_pretrained(path_labse)
        self.tokenizer_labse = AutoTokenizer.from_pretrained(path_labse)
        self.stopwords_russian = stopwords.words('russian')
        # self.path_embedding = path_embedding

    def embed_bert_cls(self,
                       text):
        t = self.tokenizer_labse(
            text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model_labse(
                **{k: v.to(self.model_labse.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    def read_csv_news(self, path_data):
        self.data_news = pd.read_csv(path_data, sep=';')
        return self.data_news

    def read_embeddings_by_user(self, path):
        with open(path, 'rb') as f:
            matrix_emb_user = np.load(f)
        return matrix_emb_user

    def create_matrix_interest_by_user(self, path_data, path_user):

        data = self.read_csv_news(path_data)
        len_date = len(data)
        mean_emb = self.read_embeddings_by_user(path_user)
        matrix = np.zeros((len_date, 770), dtype=np.float32)
        data['lem_token'] = np.nan
        for cur in range(len_date):
            sentence = self.remove_stop_words(data['title'][cur])
            tmp_sentence = self.lemmatize_text(sentence)
            data['lem_token'][cur] = tmp_sentence

        for i in trange(len_date):
            matrix[i, 0] = i
            matrix[i, 2:] = np.array(self.embed_bert_cls(
                data['lem_token'][i]), dtype=np.float32)

        for count in range(matrix.shape[0]):
            matrix[count][1] = cosine_similarity_numba(mean_emb,
                                                       matrix[count][2:])

        return matrix, (matrix[matrix[:, 1].argsort()][::-1])[:3]

    def end_news_link(self, path_n, path_u):

        matrix, data = self.create_matrix_interest_by_user(path_n, path_u)
        data_not_touch = self.read_csv_news(path_n)
        results = []

        for i in range(3):
            results.append(
                {data_not_touch['title'][data[i][0]]: data_not_touch['link'][data[i][0]]})

        return results

    def remove_stop_words(self, text):
        tokens = word_tokenize(text)
        tokens = [
            token for token in tokens if token not in self.stopwords_russian and token != ' ']
        return " ".join(tokens)

    def lemmatize_text(self, text):
        words = text.split()
        res = []
        for word in words:
            p = morph.parse(word)[0]
            res.append(p.normal_form)

        return " ".join(x for x in res)
