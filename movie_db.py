import http.client
import json
import random


def setup():
    conn = http.client.HTTPSConnection("api.themoviedb.org")

    payload = "{}"

    conn.request("GET", "/3/movie/popular?page=1&language=en-US&api_key=54d152b4b8b5fc95233effb724e9f1a0", payload)

    res = conn.getresponse()
    data = json.loads(res.read())
    # pprint(data['results'])

    conn.request("GET", "/3/genre/movie/list?language=en-US&api_key=54d152b4b8b5fc95233effb724e9f1a0", payload)

    res = conn.getresponse()

    genre = json.loads(res.read().decode("utf-8"))
    # pprint(genre)
    return data, genre


data, data_genre = setup()


def get_genre(id):
    for i in range(len(data_genre['genres'])):
        if id == data_genre['genres'][i]['id']:
            return data_genre['genres'][i]['name']


# facts
title_labels = ["The movie title is {}.", "The movie is entitled {}.", "The movie is {}."]
rating_pos = ["The movie is good.", "The movie is amazing.", "I have seen a great movie."]
rating_neg = ["The movie is bad.", "Worst movie ever.", "I have seen better movies."]
rating_val = ["I would rate {}.", "The movie has {}.", "The movie is marked as {}."]
release_val = ["Released in {}.", "Dating from {}.", "Premiered {}.", "First on screen {}."]
release_old = ["The movie is old.", "Ancient movie.", "The movie is retro."]
release_new = ["The movie is new.", "Fresh movie.", "Just hit the cinema."]
popularity_val = ["Popularity {}.", "The movie has been seen by {} people.", "The cinema was {} filled."]
popular = ["The movie is very popular.", "The movie is well known.", "Many people have seen this movie."]
unpopular = ["Didn't know about this movie.", "Haven't seen it yet.", "This movie is less popular."]
genre_val = ["This is a {}.", "The movie is about {}.", "The movie falls in {} category.", "The genre of movie is {}."]
# questions
question_genre = ["What is this movie about?\t{}\t{}", "What genre the movie falls in?\t{}\t{}",
                  "About what is the movie?\t{}\t{}"]
question_popularity = ["Is this movie popular?\t{}\t{}", "How many people like this movie?\t{}\t{}",
                       "Is it popular?\t{}\t{}"]
question_rating = ["Is it a good movie?\t{}\t{}", "How much was it rated?\t{}\t{}",
                   "What rating does the movie have?\t{}\t{}"]
question_title = ["What is the movie name?\t{}\t{}", "Do you know the name of movie?\t{}\t{}",
                  "Can you tell me the name of movie?\t{}\t{}", "Which movie was it?\t{}\t{}"]


def random_list(label):
    return random.choices(label, k=len(label))


def generator(row):
    row = list(row)
    rowe = []
    for i in range(len(row)):
        row[i] = str(row[i])
        rowe.append([])
        for j in range(len(row[i])):
            rowe[i].append(row[i][j])
            if row[i][j] == "\'":
                rowe[i].append("\'")
    row = [''.join(rowe[i]) for i in range(len(rowe))]

    string = "".join(''.format(k) for k in row)
    return string


def word_chck(word):
    if "\'" in word:
        # print(word)
        word = generator(word)
    return word


with open('movie_train.txt', "w", encoding="utf8") as movie_train:
    for movie in data['results']:
        n = 1
        title = word_chck(movie['title'])
        overview = word_chck(movie['overview'])
        adult = movie['adult']
        vote_average = movie['vote_average']
        release_date = int(movie['release_date'][0:4])
        popularity = int(float(movie['popularity']) * 1000)
        genres = [get_genre(genre) for genre in movie['genre_ids']]
        movie_train.write(str(n) + " " + random.choice(title_labels).format(title) + "\n")
        n += 1
        vote_n = n
        if vote_average > 6:
            vote_ans = "great"
            movie_train.write(str(n) + " " + random.choice(rating_pos) + "\n")
        else:
            vote_ans = "bad"
            movie_train.write(str(n) + " " + random.choice(rating_neg) + "\n")
        n += 1
        # movie_train.write(str(n) + " " + random.choice(rating_val).format(int(vote_average)) + "\n")
        # n += 1
        # movie_train.write(str(n) + " " + random.choice(release_val).format(release_date) + "\n")
        # n += 1
        if release_date < 2015:
            movie_train.write(str(n) + " " + random.choice(release_old) + "\n")
        else:
            movie_train.write(str(n) + " " + random.choice(release_new) + "\n")
        n += 1
        for sentence in overview.replace("...", '.').split(".")[:-1]:
            movie_train.write(str(n) + " " + sentence.strip() + ".\n")
            n += 1
        # movie_train.write(str(n) + " " + random.choice(popularity_val).format(popularity) + "\n")
        # n += 1
        pop_n = n
        if popularity > 80000:
            pop_ans = "yes"
            movie_train.write(str(n) + " " + random.choice(popular) + "\n")
        else:
            pop_ans = "no"
            movie_train.write(str(n) + " " + random.choice(unpopular) + "\n")
        n += 1
        genre_n = n
        for genre in genres:
            movie_train.write(str(n) + " " + random.choice(genre_val).format(genre) + "\n")
            n += 1
        movie_train.write(str(n) + " " + random.choice(question_genre).format(genres[0], genre_n) + "\n")
        n += 1
        movie_train.write(str(n) + " " + random.choice(question_title).format(title, 1) + "\n")
        n += 1
        movie_train.write(str(n) + " " + random.choice(question_popularity).format(pop_ans, pop_n) + "\n")
        n += 1
        movie_train.write(str(n) + " " + random.choice(question_rating).format(vote_ans, vote_n) + "\n")
