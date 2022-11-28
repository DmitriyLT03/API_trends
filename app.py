from dataclasses import dataclass
import fnmatch
import json
import os
from pathlib import Path
from flask import Flask, make_response, request
from flask_cors import CORS
from datetime import datetime, timedelta
from typing import List
import requests
from bs4 import BeautifulSoup
import time
from Trends import Trends
import nltk
from Digest import Digest
import ssl


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["get", "post"])
def index():
    return make_response("ok")


def find(pattern, path="."):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


@dataclass
class NewsItem:
    title: str
    date: datetime
    link: str

    def __str__(self) -> str:
        return f"{self.title};{self.date.strftime('%Y-%m-%d')};{self.link}\n"


def parse_date(date: datetime) -> List[NewsItem]:
    """Парсит новости за определенную дату"""
    time.sleep(1)
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "cache-control": "max-age=0",
        "sec-ch-ua": "\"Google Chrome\";v=\"105\", \"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"105\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1"
    }
    print("start parsing", date.strftime("%Y-%m-%d"))

    news: List[NewsItem] = []

    lenta_url: str = f"""https://lenta.ru/news/{date.strftime("%Y/%m/%d")}/page/1"""
    ria_url: str = f"""https://ria.ru/{date.strftime("%Y%m%d")}/"""

    response = requests.get(lenta_url, headers=headers).text
    soup = BeautifulSoup(response, features="html.parser")
    titles_ = soup.find_all(attrs={"class": "card-full-news__title"})
    links_ = soup.find_all(attrs={"class": "card-full-news"})
    titles = [t.text for t in titles_]
    links = [f"https://lenta.ru{u.attrs['href']}" for u in links_]
    for i in range(len(titles_)):
        news.append(
            NewsItem(titles[i], date, links[i])
        )

    response = requests.get(ria_url, headers=headers).text
    soup = BeautifulSoup(response, features="html.parser")
    titles_ = soup.find_all(name="a", attrs={"class": "list-item__title"})
    titles = [t.text for t in titles_]
    for i in range(len(titles_)):
        news.append(
            NewsItem(titles[i], date, titles_[i].attrs['href'])
        )

    return news


def download_news() -> Path:
    date = datetime.today().strftime('%Y%m%d')
    news_path = f"news_{date}.csv"
    if os.path.exists(news_path):
        return news_path

    news_list = find("news_*.csv")
    for n in news_list:
        os.remove(n)

    filename = datetime.today().strftime("news_%Y%m%d.csv")
    with open(filename, "wt") as f:
        f.write("title;date;link\n")
        for date in (datetime.today() - timedelta(n) for n in range(0, 7)):
            news = parse_date(date)
            for n in news:
                f.write(str(n))

    return filename


def update_trends():
    date = datetime.today().strftime('%Y-%m-%d')
    trends_path = f"rec_trends_{date}.json"
    print(trends_path)
    if os.path.exists(trends_path):
        print("FILE EXISTS")
        return trends_path
    print("DONT EXIST")
    news_list = find("rec_trends_*.json")
    # for n in news_list:
    # os.remove(n)

    # filename = datetime.today().strftime("rec_trends_%Y%m%d.json")
    filename = datetime.today().strftime("news_%Y%m%d.csv")
    t = Trends(filename,
               "LaBSE-en-ru/", "rut5-base-absum/")

    t.check_trends()
    return filename


def update_news():
    download_news()
    return update_trends()


def reparse_trends(data):
    res = []

    for k in data:
        title = list(k.keys())[0]
        links = list(k.values())[0]
        res.append(
            {
                "title": title,
                "links": links
            }
        )
    return json.dumps({"items": res}, ensure_ascii=False)


def reparse_(data):
    res = []

    for k in data:
        title = list(k.keys())[0]
        links = list(k.values())[0]
        res.append(
            {
                "title": title,
                "links": [links]
            }
        )
    return json.dumps({"items": res}, ensure_ascii=False)


def reparse(path):
    with open(path, "rt") as f:
        data = json.load(f)
    return reparse_trends(data)


@app.route("/trends", methods=["get", "post"])
def trends():
    if request.method == 'GET':
        return make_response("<h1 style='text-align: center; margin-top: 10vh;'>Сюда нельзя!</h1>")

    file = update_news()
    a = reparse(file)
    # print(a)
    response = make_response(a)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET, PUT, POST, DELETE')
    response.headers.add('Access-Control-Allow-Headers',
                         'Origin, X-Requested-With, Content-Type, Accept')

    return response


@app.route("/digest", methods=["get", "post"])
def digest():
    if request.method == 'GET':
        return make_response("<h1 style='text-align: center; margin-top: 10vh;'>Сюда нельзя!</h1>")
    role = json.loads(request.data).get("role")
    if role == 0:
        d = Digest("LaBSE-en-ru")
        a = d.end_news_link(datetime.today().strftime(
            "news_%Y%m%d.csv"), "mean_manager.npy")
        # response = make_response(json.dumps(
        #     {"items": json.loads(reparse_(a))}, ensure_ascii=False)
        # )
        response = make_response(json.loads(reparse_(a)))
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods',
                             'GET, PUT, POST, DELETE')
        response.headers.add('Access-Control-Allow-Headers',
                             'Origin, X-Requested-With, Content-Type, Accept')
        return response
    if role == 1:
        d = Digest("LaBSE-en-ru")
        a = d.end_news_link(datetime.today().strftime(
            "news_%Y%m%d.csv"), "mean_accountant.npy")
        # response = make_response(json.dumps(
        #     {"items": json.loads(reparse_(a))}, ensure_ascii=False))
        response = make_response(json.loads(reparse_(a)))
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods',
                             'GET, PUT, POST, DELETE')
        response.headers.add('Access-Control-Allow-Headers',
                             'Origin, X-Requested-With, Content-Type, Accept')
        return response

    response = make_response(json.dumps(
        {"items": []}, ensure_ascii=False))
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET, PUT, POST, DELETE')
    response.headers.add('Access-Control-Allow-Headers',
                         'Origin, X-Requested-With, Content-Type, Accept')

    return response


@app.route("/insites_old", methods=["get", "post"])
def insites_old():
    if request.method == 'GET':
        return make_response("<h1 style='text-align: center; margin-top: 10vh;'>Сюда нельзя!</h1>")

    with open("insite.json", "rt") as f:
        insites = json.load(f)

    response = make_response(insites)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET, PUT, POST, DELETE')
    response.headers.add('Access-Control-Allow-Headers',
                         'Origin, X-Requested-With, Content-Type, Accept')

    return response


@app.route("/insites", methods=["get", "post"])
def insites():
    if request.method == 'GET':
        return make_response("<h1 style='text-align: center; margin-top: 10vh;'>Сюда нельзя!</h1>")

    filename = datetime.today().strftime("news_%Y%m%d.csv")
    t = Trends(filename,
               "LaBSE-en-ru/", "rut5-base-absum/")
    _, b = t.check_trends()
    response = make_response(b)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET, PUT, POST, DELETE')
    response.headers.add('Access-Control-Allow-Headers',
                         'Origin, X-Requested-With, Content-Type, Accept')

    return response


if __name__ == "__main__":
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')
    nltk.download('punkt')
# date = datetime.today()
# file = date.strftime("news_%Y%m%d.csv")
# # fetch_news(file)
# news = parse_date(date)
# with open(file, "wt", encoding="utf-8") as f:
#     f.write("title;date;link\n")
#     for n in news:
#         f.write(str(n))
    app.run(host="0.0.0.0", port="3001")
