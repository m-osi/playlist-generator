import pandas as pd
from create_pipe import CleanText, Tokenize
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
tokenize = Tokenize(columns='lyric')
df = tokenize.transform(df)
vocab = tokenize.vocab


# saving the master dataset

joblib.dump(vocab, r".\data\vocab.pkl")