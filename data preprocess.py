import re
import pandas as pd  
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

def clean_text(text):
    text=text.lower() 
    text = re.sub(r'^\s*(start\s*article|startarticle)[\s\W]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstart\s*article\b', '', text)
    text = re.sub(r'startarticle', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_stopwords(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_tokens = [token for token in filtered_tokens if token != '\n'] 
    sorted_tokens = sorted(filtered_tokens, key=len)
    return sorted_tokens

def build_ngram_features(corpus, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngram_matrix = vectorizer.fit_transform(corpus)
    return ngram_matrix, vectorizer

def remove_start_tokens(text_or_tokens):
    """
    Removes occurrences of 'startarticle', 'start article', 'startparagraph', 'start section', and similar tokens
    from the input text or list of tokens.
    """
    start_tokens = ['startarticle', 'start article', 'startparagraph','start paragraph','startsection' 'start section', 'start', 'article', 'paragraph', 'section']
    
    if isinstance(text_or_tokens, str):
        for token in start_tokens:
            text_or_tokens = text_or_tokens.replace(token, '')
        return text_or_tokens.strip()
    elif isinstance(text_or_tokens, list):
        return [token for token in text_or_tokens if token not in start_tokens]
    else:
        return text_or_tokens


def main():
    test_dataset = load_dataset("google/wiki40b","en", split="test")
    validation_dataset = load_dataset("google/wiki40b","en", split="validation")
    print("length of test dataset:", len(test_dataset))
    print("length of validation dataset:", len(validation_dataset))
    
    test_df = pd.DataFrame(test_dataset)
    validation_df = pd.DataFrame(validation_dataset)
    
    print("Columns in test dataset:", test_df.columns)
    print("Columns in validation dataset:", validation_df.columns)
    
    if 'text' not in test_df.columns or 'text' not in validation_df.columns:
        print("The 'text' column is not present in the dataset.")
        return
    
    print("cleaning test data...")
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)
    validation_df['cleaned_text'] = validation_df['text'].apply(clean_text)
    
    print("tokenizing and removing stop words from test data...")
    test_df['tokens'] = test_df['cleaned_text'].apply(tokenize_and_stopwords)
    validation_df['tokens'] = validation_df['cleaned_text'].apply(tokenize_and_stopwords)
    
    test_df['cleaned_text'] = test_df['cleaned_text'].apply(remove_start_tokens)
    test_df['tokens'] = test_df['tokens'].apply(remove_start_tokens)
    validation_df['cleaned_text'] = validation_df['cleaned_text'].apply(remove_start_tokens)
    validation_df['tokens'] = validation_df['tokens'].apply(remove_start_tokens)
    
    print("building n-gram features for test data...")
    ngram_matrix, vectorizer = build_ngram_features(test_df['cleaned_text'], ngram_range=(1, 2))
    print("N-gram features shape:", ngram_matrix.shape)
    
    print('saving test data...')
    test_df[['cleaned_text','tokens']].to_csv('preprocessed_test.csv', index=False)
    validation_df[['cleaned_text','tokens']].to_csv('preprocessed_validation.csv', index=False)
    
    print("Test and validation data preprocessed and saved successfully.")
    
if __name__ == "__main__":
    main()
    