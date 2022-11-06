import joblib
from src.create_pipe import embedding_for_vocab
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# load pickled files
final = joblib.load(r"./data/master.pkl")
vocab = joblib.load(r"./data/vocab.pkl")
song_embeddings = joblib.load(r"./data/embeddings.pkl")

STATIC_FEATURES = ['dance_x', 'acousticness', 'valence']

def get_neighbors(n_count, dataset, feature):
    nn = NearestNeighbors(n_neighbors=n_count, algorithm='ball_tree')
    nn.fit(dataset)
    nn_results = nn.kneighbors([feature])
    nn_results = list(nn_results[1][0])
    nn_songs = [final['name_x'][i] for i in nn_results]
    return nn_songs

def common_elements(list1, list2):
    return [element for element in list1 if element in list2]

def get_embedding(word, embedding_path=r"./data/glove/glove.6B.50d.txt"):
    word_dict = {word: 1}
    embedding = embedding_for_vocab(
        embedding_path,word_dict,50)
    if np.all(embedding[1]==0):
        message = f"Main lyrics theme not supported"
        raise Exception(message)
    return embedding

def get_top_words(n_count, embedding, dataset):
    nn = NearestNeighbors(
        n_neighbors=n_count, 
        algorithm='ball_tree', 
        metric='euclidean')
    nn.fit(dataset)
    predicted_words = nn.kneighbors(embedding[[1]])
    predicted_words_results = list(predicted_words[1][0])
    top_words = [list(
        vocab.keys())[list(
            vocab.values()).index(x)] 
            for x in predicted_words_results]
    return top_words


def predict(example, vectors):
    example = example.lower()
    embedding = get_embedding(example)
    top_words = get_top_words(n_count=4, embedding=embedding, dataset=song_embeddings)
    features = STATIC_FEATURES + top_words
    try:
        with_words = final[features]
    except Exception:
        message = f"Main lyrics theme not supported"
        raise Exception(message)
    with_words['word_mean'] = 0.6 * with_words.iloc[:, 3] \
        + 0.2 * with_words.iloc[:, 4] \
            + 0.15 * with_words.iloc[:, 5] \
                + 0.05 * with_words.iloc[:, 6]
    features_with_word_mean = STATIC_FEATURES + ['word_mean']
    with_words = with_words[
        features_with_word_mean]

    scaler = MinMaxScaler()
    for feature in list(with_words.columns):
        with_words[[feature]] = scaler.fit_transform(
            with_words[[feature]])

    all_features_songs = get_neighbors(
        n_count=20, 
        dataset=with_words.values, 
        feature=vectors)
    nn_words_songs = get_neighbors(
        n_count=30, 
        dataset=with_words[['word_mean']].values, 
        feature=[1])
    songs_fin = common_elements(
        all_features_songs, nn_words_songs)

    songs_no_words = [x for x in all_features_songs \
        if x not in songs_fin]

    if len(songs_fin) == 10:
            songs_10 = songs_fin
    elif len(songs_fin) > 10:
        songs_10 = songs_fin[:10]
    else:
        songs_10 = songs_fin + songs_no_words[:10-len(songs_fin)]
    return songs_10


if __name__ == "__main__":
    playlist = predict('cat', [0.6,0.3,0.2,1])
    print(playlist)













