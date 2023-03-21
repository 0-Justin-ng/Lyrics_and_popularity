# Lyrics and Spotify Popularity

This project will attempt to predict Spotify popularity based on views using various representations of song lyrics. 
The dataset can be found [here](https://www.cs.cornell.edu/~arb/data/genius-expertise/).

# Weekly Goals
- [x] Reduce vocabulary size.
- [x] Run linear regressiWon and lasso model with reduced vocabulary size.
- [ ] Set up thresholds for popularity (high, medium and low popularity) for logistic regression.
- [ ] Add disclaimer for explicit words. 
- [ ] Look into joblib to store data. 
    - Stores complicated python objects.
    - This only saves model.
- [ ] Look into parquet.
    - Saves your data.
    - Save output of count vectorizer.  
- [ ] Look into grid search
- [ ] Look into pipeline
- [ ] Look into a binary cutoff. 
- [ ] Look into n-Grams for lyrics transformations.
- [ ] Non-linear classifiers
    - [ ] Random Forest: max depth = 4 
- [ ] Obsidian - for notes. 