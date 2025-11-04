# Enhancing Social Media Friend Recommendations through User Post Analysis

## Project Overview
This project explores optimizing friend recommendation systems on social media platforms by creating a comprehensive user similarity metric based on users' post content. The system analyzes tweet data to determine if user posts can effectively reflect similarity between users for improved friend recommendations.

## Team Members
- Dvir Shitrit
- Roey Fabian
- Danielle Eldor
- Guy Kalati

## Problem Definition
The project aims to solve the optimization of friend recommendation systems on social media platforms by creating a more comprehensive user similarity metric to enhance friend recommendation algorithms based on users' common denominators, specifically through analysis of their post content.

## Dataset Description

### Data Sources
The project uses three Twitter datasets from Kaggle:

1. **COVID-19 Tweets Dataset**
   - Contains tweets with #covid19 hashtag
   - 92,276 rows, 13 columns
   - Collected via Twitter API

2. **Australian Election 2019 Tweets Dataset**
   - Tweets from 2019 Australian federal election
   - 182,748 rows, 11 columns
   - Collected via Twitter API

3. **Tokyo Olympics 2020 Tweets Dataset**
   - Tweets with #Tokyo2020 hashtag
   - 63,928 rows, 16 columns
   - Collected via Twitter API

### Combined Dataset Structure
- **Total initial size**: 397,596 rows
- **Final cleaned dataset**: 20,377 rows, 1,000 unique users
- **Columns**: 
  - `user_name`: Username of account
  - `user_description`: User bio/description
  - `user_location`: User location information
  - `text`: Tweet content
  - `label`: Source dataset (covid/australian/tokyo)
  - `id`: Unique user identifier

### Dataset Distribution
- After cleaning:
  - Australian: 37.31%
  - Tokyo: 31.61%
  - Covid: 31.08%

## Project Structure

### Notebooks
- `cleaning_blog_data_text_and_post_features.ipynb` - Main data cleaning and feature engineering
- `cleaning_blog_data_description_and_user_features.ipynb` - User description analysis
- `COVID19_attempted_clustering_TFIDF_and_user_data.ipynb` - Clustering experiments with TF-IDF

## Methodology

### 1. Data Cleaning Process
- **Missing value handling**: Removed rows with null values using dropna() and regex for empty strings
- **Duplicate removal**: Handled at user and post level (ID-text pairs)
- **User sampling**: Retained users in 95th percentile of post counts
- **Balanced sampling**: Applied weighted sampling technique to balance dataset distribution

### 2. Feature Engineering
#### Implemented Features:
- **User Description Tokenization**: Tokenized user descriptions and removed stop words
- **Text Tokenization**: Tokenized post texts and removed stop words
- **Aggregated User Vectors**: Grouped multiple posts per user into aggregated embeddings
- **User IDs**: Created unique IDs based on username, description, and location combination

#### Features Considered but Not Used:
- Hashtag embeddings and count (already captured in Doc2Vec)
- User ID by username and date created (replaced with more comprehensive ID system)

### 3. Models Implemented

#### Doc2Vec (Gensim)
- Parameters: `vector_size=1000, window=3, min_count=1, workers=4, epochs=35`
- Captures document-level semantics and word order

#### SBERT (Sentence-BERT)
- Pre-trained model: `paraphrase-MiniLM-L3-v2`
- Zero-shot approach (no fine-tuning)
- Excels at sentence-level semantic similarity

#### LDA (Latent Dirichlet Allocation)
- Parameters: `num_topics=30, random_state=100, chunksize=100, passes=10`
- Discovers latent topics in user content

### 4. Evaluation Approach
- Used dataset labels as golden standard
- Compared similarity between users from same vs different datasets
- Evaluated top-k (k=30) recommendations
- Metrics: Jaccard Similarity, Kendall's Tau, NDCG

## Key Findings

1. **Model Performance**: SBERT showed the best performance in identifying friendship recommendations across all dataset sizes
2. **Content Analysis**: Post texts successfully identified potential friendship recommendations
3. **User Description vs Posts**: Limited correspondence between post content and user bio content, revealing significant distance between these two information sources
4. **Label Similarity Metrics**:
   - 1,000-user dataset achieved consistent performance across models
   - 5,000-user dataset maintained effectiveness with larger scale

## Requirements

```python
# Core libraries
pandas
numpy
nltk
scikit-learn
gensim

# For Doc2Vec
gensim.models.doc2vec

# For SBERT
sentence-transformers

# For visualizations
matplotlib
seaborn
```

## Usage

1. **Data Collection**: Download the three Twitter datasets from Kaggle
2. **Data Preprocessing**: Run the cleaning notebooks to prepare the combined dataset
3. **Feature Engineering**: Execute feature extraction and tokenization
4. **Model Training**: Train Doc2Vec, SBERT, and LDA models on the processed data
5. **Evaluation**: Generate similarity matrices and evaluate recommendations

## Future Improvements

- **Scaling to Real Social Networks**: Integration with live platforms (Facebook, Twitter, LinkedIn)
- **Real-Time Data Streams**: Incorporate dynamic updates based on user activities
- **Enhanced Model Complexity**: Explore advanced transformer architectures and ensemble methods
- **Privacy Considerations**: Develop transparent, user-centric recommendation systems
- **Real-World Validation**: Conduct pilot studies with actual social media platforms

## Limitations

- Dataset limited to Twitter data from specific events (COVID-19, Australian election, Tokyo Olympics)
- Sample size reduced to 1,000 users for computational feasibility
- No actual friendship ground truth data (used dataset labels as proxy)
- Static analysis without temporal dynamics

## License
This project uses publicly available datasets from Kaggle.

## Contact
For questions or collaborations, please contact the team members through the provided student IDs.
