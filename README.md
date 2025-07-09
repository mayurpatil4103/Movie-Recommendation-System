# Movie-Recommendation-System

### **🔍 Objective:**
**Movie Recommendation System using Python and scikit-learn that recommends similar movies based on the analysis of descriptive features such as genres, keywords, tagline, cast, and director. By leveraging TF-IDF vectorization and cosine similarity, the system finds and suggests movies that are most similar in content to the one entered by the user, helping them discover new movies in a smart and meaningful way.**

---
### **1. Importing Libraries**
```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
- `pandas` is used for data manipulation and analysis.
- `numpy` is used for numerical operations.
- `difflib` helps to finding close matches to user input.
- `sklearn.feature_extraction.text` helps to converting text to numerical vectors.
- `sklearn.metrics.pairwis` is used for computing similarity between movies.

---
### **2. Load Movie Data**
```python
movies_data = pd.read_csv('movies.csv')
```
- Reads the dataset from a CSV file into a Pandas DataFrame.

---
### **3. Exploring the Dataset**
```python
print(movies_data.head())
print(movies_data.shape)
print(movies_data.info())
```
- Provides information about the dataset, including column names, data types, and non-null values.

---
## **4. Select and Prepare Important Features**
```python
features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in features:
    movies_data[feature] = movies_data[feature].fillna('')  # Replace missing values with empty strings

combined = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

```
- select key movie details and replace missing data with empty text. Then, we combine these details into one text string for each movie to help compare them easily.

---
## **5. Convert Text Data To Numerical Feature**
```python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined)

```
- `TfidfVectorizer` converts the combined text into numeric vectors that reflect the importance of each word across all movies. This helps capture the movie’s content in a way computers can understand.
---
## **6. Calculate Similarity Score**
```python
similarity = cosine_similarity(feature_vectors)
print(f"Similarity matrix shape: {similarity.shape}")

```
- `cosine_similarity` computes how similar each movie vector is to every other movie vector. The output is a matrix where each cell represents the similarity score between two movies.
---
## **7. User Input & Match Search**
```python
user_movie = input('Enter your favourite movie name: ')
all_titles = movies_data['title'].tolist()
matches = difflib.get_close_matches(user_movie, all_titles)

if not matches:
    print("Sorry, no matching movie found.")
else:
    matched_title = matches[0]
    print(f"Did you mean: {matched_title}?")

```
- Since users might type movie names with typos or variations, `difflib.get_close_matches` helps find the closest movie title in the dataset to what the user entered. If no close match is found, it informs the user.
---
## **8. Find Similar Movies**
```python
movie_index = movies_data[movies_data.title == matched_title]['index'].values[0]
movie_scores = list(enumerate(similarity[movie_index]))
sorted_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)

```
- Once the exact movie is identified, we pull all similarity scores for that movie compared to every other movie. These scores are sorted from highest to lowest similarity.
---
## **9. Display Top Movie Recommendation**
```python
print("Here are some movies you might like:\n")
recommended = []
count = 1

for idx, score in sorted_movies:
    title = movies_data[movies_data.index == idx]['title'].values[0]
    if count < 30:
        print(f"{count}. {title}")
        recommended.append(title)
        count += 1

```
- The program lists the top 29 movies most similar to the user’s favorite movie, based on content similarity.
---
## **10. Save Recommendation To CSV**
```python
recommendation_df = pd.DataFrame(recommended, columns=['Recommended Movies'])
recommendation_df.to_csv('recommended_movies.csv', index=False)
```
- Saves the recommended movie list as a CSV file so users can keep or share the suggestions easily.
---


