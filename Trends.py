from pickletools import floatnl
import pandas as pd
import numpy as np
import torch
import os
import datetime
import json
import pymorphy2
from tqdm import tqdm
from tqdm import trange
from importlib.resources import path
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from transformers import T5ForConditionalGeneration, T5Tokenizer
from numba import jit
from nltk.corpus import stopwords
from nltk import word_tokenize

morph = pymorphy2.MorphAnalyzer()


@jit(nopython=True, cache=True)
def euclidean_distance(u: np.ndarray, v: np.ndarray) -> np.float32:
    return np.linalg.norm(u - v)


@jit(nopython=True, cache=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray) -> np.float64:
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 1.
    return np.dot(u, v)


class Trends:

    def __init__(self, path_news: path, path_model_labse: path, path_model_rut5: path):
        self.stopwords_russian = stopwords.words('russian')
        self.new_path_trends = ''
        self.new_path_embed = ''
        self.new_path_insite_embeddings = ''
        self.new_path_insite = ''
        self.path_insite_json = './insite.json'
        self.data_news = pd.read_csv(path_news, sep=';')
        self.tokenizer_labse = AutoTokenizer.from_pretrained(path_model_labse)
        self.model_labse = AutoModel.from_pretrained(path_model_labse)
        self.tokenizer__rut5 = T5Tokenizer.from_pretrained(path_model_rut5)
        self.model__rut5 = T5ForConditionalGeneration.from_pretrained(
            path_model_rut5)

    def cheack_date(self):
        previous_date = datetime.datetime.today()
        path_trends = './rec_trends' + '_' + \
            previous_date.strftime("%Y-%m-%d") + '.json'
        path_embed = './trends_embedding' + '_' + \
            datetime.datetime.now().strftime("%Y-%m-%d") + '.npy'
        path_insite_embed = './insite_embeddings' + '_' + \
            datetime.datetime.now().strftime("%Y-%m-%d") + '.npy'
        path_insite_text = './insite_text' + '_' + \
            datetime.datetime.now().strftime("%Y-%m-%d") + '.json'
        return path_trends, path_embed, path_insite_embed, path_insite_text

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

    def check_trends(self):
        self.new_path_trends, self.new_path_embed, self.new_path_insite_embeddings, self.new_path_insite = self.cheack_date()
        if os.path.exists(self.new_path_trends) and os.path.exists(self.new_path_insite):
            with open(self.new_path_trends, 'rb') as f:
                file_trend = json.load(f)
            with open(self.new_path_insite, 'rb') as f:
                file_insite = json.load(f)
            return file_trend, file_insite
        else:
            self.sentences = self.get_sentences()
            self.link = self.get_link()
            if os.path.exists(self.new_path_embed):
                with open(self.new_path_embed, 'rb') as f:
                    self.trends_embedding = np.load(f)
            else:
                self.trends_embedding = self.create_matrix_embedding()
            if os.path.exists(self.new_path_insite_embeddings):
                with open(self.new_path_insite_embeddings, 'rb') as f:
                    self.insite_embeddings = np.load(f)
            else:
                self.insite_embeddings = self.create_matrix_insite(
                    self.path_insite_json)
            self.opt_k = self.find_optimum_k_means()
            self.trained_k_means = self.train_opt_k_means()
            self.topic_text, self.topic_link = self.search_nearest_centences()
            return self.create_trands(self.new_path_insite_embeddings, self.path_insite_json)

    def get_sentences(self):
        return list(self.data_news['title'])

    def get_link(self):
        return list(self.data_news['link'])

    def create_matrix_embedding(self):
        matrix_embedding = np.zeros(
            (len(self.data_news), 768), dtype=np.float32)
        self.list_sentences = self.get_sentences()
        for count in trange(len(self.list_sentences)):
            sentence = self.remove_stop_words(self.list_sentences[count])
            tmp_sentence = self.lemmatize_text(sentence)
            micro_result = tmp_sentence
            matrix_embedding[count] = np.array(
                self.embed_bert_cls(micro_result), dtype=np.float32)
        np.save(self.new_path_embed, matrix_embedding)
        return matrix_embedding

    def find_optimum_k_means(self):
        def calc_distance(x1, y1, a, b, c):
            dist = abs((a * x1 + b * y1 + c)) / (np.sqrt(a * a + b * b))
            return dist
        k_min = 6
        num_clusters = np.array([x for x in range(2, k_min * 5)])  # K
        calculate_clusters = []  # dist_points
        for count in trange(len(num_clusters)):
            calculate_clusters.append(
                (KMeans(n_clusters=num_clusters[count]).fit(self.trends_embedding)).inertia_)
        coef_a = calculate_clusters[0] - calculate_clusters[-1]
        coef_b = num_clusters[-1] - num_clusters[0]
        coef_div_c = num_clusters[0] * calculate_clusters[-1] - \
            num_clusters[-1] * calculate_clusters[0]
        calculate_new_clusters = []
        for knum in range(len(num_clusters)):
            calculate_new_clusters.append(calc_distance(num_clusters[knum],
                                                        calculate_clusters[knum],
                                                        coef_a,
                                                        coef_b,
                                                        coef_div_c))
        return np.argmax(np.array(calculate_new_clusters))

    def train_opt_k_means(self):
        kmeans = KMeans(n_clusters=self.opt_k, random_state=42).fit(
            self.trends_embedding)
        return kmeans

    def search_nearest_centences(self):
        data = pd.DataFrame()
        data['text'] = self.sentences
        data['link'] = self.link
        data['label'] = self.trained_k_means.labels_
        data['embedding'] = list(self.trends_embedding)

        kmeans_centers = self.trained_k_means.cluster_centers_
        top_texts_list = []
        top_link_list = []
        for count in range(self.opt_k):
            cluster = data[data['label'] == count]
            embeddings = list(cluster['embedding'])
            texts = list(cluster['text'])
            link = list(cluster['link'])
            dist = [euclidean_distance(kmeans_centers[0].reshape(
                1, -1), emb.reshape(1, -1)) for emb in embeddings]
            scores = list(zip(texts, link, dist))
            top_3 = sorted(scores, key=lambda x: x[2])[:3]
            top_texts = list(zip(*top_3))[0]
            top_texts_list.append(top_texts)
            top_link = list(zip(*top_3))[1]
            top_link_list.append(top_link)
        return top_texts_list, top_link_list

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

    def summarize(self,
                  text,
                  n_words=None,
                  compression=None,
                  max_length=1000,
                  num_beams=3,
                  do_sample=False,
                  repetition_penalty=10.0,
                  **kwargs):
        if n_words:
            text = '[{}] '.format(n_words) + text
        elif compression:
            text = '[{0:.1g}] '.format(compression) + text
        x = self.tokenizer__rut5(text, return_tensors='pt', padding=True).to(
            self.model__rut5.device)
        with torch.inference_mode():
            out = self.model__rut5.generate(
                **x,
                max_length=max_length, num_beams=num_beams,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
                **kwargs
            )
        return self.tokenizer__rut5.decode(out[0], skip_special_tokens=True)

    def create_trands(self, path_insite_embedding, path_insite_text):

        results = []
        text_insite_index_percent = []

        if os.path.exists(path_insite_embedding):
            with open(path_insite_embedding, 'rb') as f:
                matrix_insite_embedding = np.load(f)
        else:
            matrix_insite_embedding = self.create_matrix_insite(
                path_insite_text)

        if os.path.exists(path_insite_text):
            with open(path_insite_text, 'rb') as f:
                insite_text = json.load(f)['items']

        for count in range(len(self.topic_link)):
            text = self.summarize(' '.join(list(self.topic_text[count])))
            index, percent = self.search_index_insite(
                text, matrix_insite_embedding)
            list_topic = list(self.topic_link[count])
            results.append({text: list_topic})
            text_insite_index_percent.append(
                {float(percent): insite_text[int(index)]})

        with open(self.new_path_trends, 'wt', encoding='utf8') as f:
            json.dump(results, f, ensure_ascii=False)

        with open(self.new_path_trends, 'wt', encoding='utf8') as f:
            json.dump(text_insite_index_percent, f, ensure_ascii=False)

        return results, text_insite_index_percent

    def create_matrix_insite(self, path_file):
        with open('./insite.json', 'rb') as f:
            file = json.load(f)["items"]
        matrix = np.zeros((len(file), 770), dtype=np.float32)
        for i, key in enumerate(file):
            sentence = self.remove_stop_words(key['hypothesis'])
            tmp_sentence = self.lemmatize_text(sentence)
            micro_result = tmp_sentence
            matrix[i][0] = i
            matrix[i][2:] = np.array(self.embed_bert_cls(micro_result))
        np.save(self.new_path_insite_embeddings, matrix)
        return matrix

    def search_index_insite(self, text, matr):
        query_vec = np.array(self.embed_bert_cls(text), dtype=np.float32)
        for i in range(matr.shape[0]):
            matr[i][1] = cosine_similarity_numba(
                query_vec,
                matr[i][2:]
            )
        return matr[matr[:, 1].argsort()][::-1][0][0], matr[i][1]
