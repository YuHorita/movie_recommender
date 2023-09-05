import streamlit as st
import pandas as pd
import numpy as np
import gensim

st.title('映画レコメンド')

# 映画情報の読み込み
movies = pd.read_csv("data/movies.tsv", sep="\t")

# 学習済みのitem2vecモデルの読み込み
model = gensim.models.word2vec.Word2Vec.load("data/item2vec.model")

# 映画IDとタイトルを辞書型に変換
movie_titles = movies["title"].tolist()
movie_ids = movies["movie_id"].tolist()
movie_id_to_title = dict(zip(movie_ids, movie_titles))
movie_title_to_id = dict(zip(movie_titles, movie_ids))
movie_id_to_genre = dict(zip(movie_ids, movies["genre"]))
movie_id_to_tag = dict(zip(movie_ids, movies["tag"]))

st.markdown("## 1本の映画に対して似ている映画を表示する")
selected_movie = st.selectbox("映画を選んでください", movie_titles)
selected_movie_id = movie_title_to_id[selected_movie]
st.write(f"あなたが選択した映画は{selected_movie}(id={selected_movie_id})です")

# 似ている映画を表示
st.markdown(f"### {selected_movie}に似ている映画")
results = []
for movie_id, score in model.wv.most_similar(selected_movie_id, topn=3):
    title = movie_id_to_title[movie_id]
    genre = movie_id_to_genre[movie_id]
    tag = movie_id_to_tag[movie_id]
    results.append({"movie_id": movie_id, "score": score, "title": title,
                   "genre": eval(genre), "tag": eval(tag)})
results = pd.DataFrame(results)
st.write(results)


st.markdown("## 複数の映画を選んでおすすめの映画を表示する")

selected_movies = st.multiselect("映画を複数選んでください", movie_titles)
selected_movie_ids = [movie_title_to_id[movie] for movie in selected_movies]
vectors = [model.wv.get_vector(movie_id) for movie_id in selected_movie_ids]
if len(selected_movies) > 0:
    user_vector = np.mean(vectors, axis=0)
    st.markdown(f"### おすすめの映画")
    recommend_results = []
    for movie_id, score in model.wv.most_similar(user_vector):
        title = movie_id_to_title[movie_id]
        recommend_results.append(
            {"movie_id": movie_id, "title": title, "score": score})
    recommend_results = pd.DataFrame(recommend_results)
    st.write(recommend_results)
