import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from create_pipe import CleanText
import joblib

#creating lyrics dataset

folklore = pd.read_csv(r'.\data\lyrics\08-folklore_deluxe_version.csv')
evermore = pd.read_csv(r'.\data\lyrics\09-evermore_deluxe_version.csv')
lover = pd.read_csv(r'.\data\lyrics\07-lover.csv')
reputation = pd.read_csv(r'.\data\lyrics\06-reputation.csv')
the1989 = pd.read_csv(r'.\data\lyrics\05-1989_deluxe.csv')
red = pd.read_csv(r'.\data\lyrics\04-red_deluxe_edition.csv')
speak = pd.read_csv(r'.\data\lyrics\03-speak_now_deluxe_package.csv')
fearless = pd.read_csv(r'.\data\lyrics\02-fearless_taylors_version.csv')
taylor = pd.read_csv(r'.\data\lyrics\01-taylor_swift.csv')
midnights = pd.read_csv(r'.\data\lyrics\10-midnights.csv', sep=';')

df = pd.concat([
    folklore, 
    evermore, 
    lover,
    reputation,
    the1989,
    red,
    speak,
    fearless,
    taylor], 
    axis=0)

df = df[['track_title', 'lyric']]
df = pd.concat([df, midnights], axis = 0)
clean = CleanText(columns='lyric')
df = clean.transform(df)
grouped = df[[
    'track_title', 'lyric']].groupby(
        'track_title').agg(list).reset_index()
grouped['lyric'] = grouped['lyric'].apply(
    lambda x: ' '.join(x))

# get tf-idf of words for songs
vectorizer = TfidfVectorizer(
    analyzer='word', 
    stop_words=None,
    ngram_range = (1,1), 
    use_idf=True, 
    smooth_idf=True
    )

tfidf_matrix  = vectorizer.fit_transform(grouped['lyric'].tolist() )
tfidf_features = sorted(vectorizer.get_feature_names())
df_tfidf = pd.DataFrame(
    data = tfidf_matrix.toarray(),index = range(
        len(grouped)),columns = tfidf_features)
result = pd.concat(
    [grouped, df_tfidf], axis=1, join="inner")

# creating spotify metadata dataset

metadata = pd.read_csv(r'.\data\spotify\spotify_taylorswift.csv')
metadata = metadata[
    ['name', 'danceability', 'acousticness', 
    'energy', 'liveness', 'valence']]
metadata_mid = pd.read_csv(r'.\data\spotify\midnights-spotify.csv')
final_metadata = pd.concat([metadata, metadata_mid], axis=0)
final_metadata['dance'] = 0.5 * final_metadata[
    'danceability'] + 0.25 * final_metadata[
        'energy'] + 0.25 * final_metadata[
            'liveness']
final_metadata = final_metadata[[
    'name',
    'dance', 
    'acousticness', 
    'valence']]

# creating master dataframe

final = pd.merge(
    final_metadata,
    result,
    how="inner",
    on=None,
    left_on="name",
    right_on="track_title",
    left_index=False,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)

# saving the master dataset

joblib.dump(final, r".\data\master.pkl")