# Taylor Swift Playlist Generator

<img src="web/img/tay.gif">

Ever wanted to create a Taylor Swift playlist with the help of machine learning? Now you have the chance! Taylor Swift Playlist Generator is a web application designed to put together a list of 10 songs that best match the criteria you provide, namely:
- **to dance to** - the higher the rating, the more likely you are to
                    stand up and dance! High scores are perfect if you're throwing a mini
                    Swiftie party in your room.
- **to cheer me up** - opt for higher scores for happy, uplifting beats. On the contrary, if you need sad tunes to cry to, set this parameter as low as you can (and preferably grab a blanket and your favorite ice cream).
- **with an acoustic vibe** - the higher the score, the more acoustic the tunes. This one is pretty self-explanatory!
- **main lyrics theme** - give Taylor something to sing about by filling in this field with a keyword!

## Architecture
- **Tech stack** - the backend was created in Python and combined with the frontend with the help of Flask. 
- **Data** - the datasets consist of: lyrics and Spotify metadata files for all albums up to and including *evermore* downloaded from Kaggle, and additional *Midnights* lyrics and Spotify metadata files created by me. 
**Algorithm** - all lyrics were preprocessed and transformed into GloVe embeddings. Additionally, TF-IDF was calculated for all word tokens for each song. Top 4 theme words are chosen by KNN algorithm based on the main lyrics theme keyword provided by the user. Then, one word feature is calculated for each song based on the TF-IDF scores of the top 4 theme words. Next, top songs are chosen by KNN algorithm based on all user's features, which are then combined with top songs chosen only on the basis of the word feature alone. A set of 10 songs is presented taking the first (up to 10) songs found in both rankings, while the remaining ones (if any) are taken from the all features list only.

## Installation

## Credits


