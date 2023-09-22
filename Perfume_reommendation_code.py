import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Step 1: Load and preprocess the dataset
dataset = pd.read_csv("C:/Users/Student/OneDrive - King's College London/Documents/Admin/Perfume Recommendation Program/final_perfume_data.csv", encoding= 'unicode_escape')
# Perform any necessary data cleaning and preprocessing here
# Step 2: Feature extraction
features = dataset['Description'] + ' ' + dataset['Notes']  # Combine description and notes as features
# Step 3: Vectorize the features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(features.values.astype('U'))
# Step 4: Calculate similarity
similarity_matrix = cosine_similarity(feature_vectors, feature_vectors)
# Step 5: Recommend perfumes
def recommend_perfumes(liked_perfumes, top_n):
    # Find the indices of perfumes that contain the liked perfume substrings
    liked_indices = dataset[dataset['Name'].str.contains('|'.join(liked_perfumes), case=False, regex=True)].index

    # Calculate the similarity scores for the liked perfumes
    similarity_scores = similarity_matrix[liked_indices]

    # Calculate the average similarity scores, excluding the liked perfumes
    average_similarity = similarity_scores.mean(axis=0)
    average_similarity[liked_indices] = 0  # Set similarity scores of liked perfumes to 0

    # Get the indices of top n similar perfumes
    top_indices = average_similarity.argsort()[::-1][:top_n]

    # Get the names of recommended perfumes
    recommended_perfumes = dataset.loc[top_indices, 'Name'].tolist()

    return recommended_perfumes

# Prompt the user for their liked perfumes
liked_perfumes = []
for i in range(3):
    perfume = input(f"Enter the name of perfume {i+1}: ")
    liked_perfumes.append(perfume)

top_n = int(input("How many recommendations would you like?: "))

recommended = recommend_perfumes(liked_perfumes, top_n)

print("Recommended perfumes:")
for perfume in recommended:
    print(perfume)