import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def load_bbc_dataset(base_path):
    """
    Load BBC News articles and summaries from the dataset
    
    Args:
        base_path (str): Path to the BBC News Summary folder
        
    Returns:
        dict: Dictionary containing articles and summaries by category
    """
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    dataset = {}
    
    def read_file(file_path):
        """Helper function to try different encodings"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Failed to decode {file_path} with any encoding")

    for category in categories:
        dataset[category] = {
            'articles': {},
            'summaries': {}
        }
        
        # Load articles
        articles_path = os.path.join(base_path, 'News-Articles', category)
        if os.path.exists(articles_path):
            for filename in os.listdir(articles_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(articles_path, filename)
                    content = read_file(file_path)
                    article_id = filename.replace('.txt', '')
                    dataset[category]['articles'][article_id] = content

        # Load summaries
        summaries_path = os.path.join(base_path, 'Summaries', category)
        if os.path.exists(summaries_path):
            for filename in os.listdir(summaries_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(summaries_path, filename)
                    content = read_file(file_path)
                    summary_id = filename.replace('.txt', '')
                    dataset[category]['summaries'][summary_id] = content
                        
    return dataset

class BBCNewsDataset(Dataset):
    def __init__(self, base_path, tokenizer_name="facebook/bart-base", max_length=512):
        """
        Initialize BBC News Dataset
        
        Args:
            base_path (str): Path to BBC News Summary folder
            tokenizer_name (str): Name of the pretrained tokenizer
            max_length (int): Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Load the raw dataset
        self.dataset = load_bbc_dataset(base_path)
        
        # Flatten the dataset structure
        self.articles = []
        self.summaries = []
        
        for category in self.dataset:
            for article_id in self.dataset[category]['articles']:
                if article_id in self.dataset[category]['summaries']:
                    self.articles.append(self.dataset[category]['articles'][article_id])
                    self.summaries.append(self.dataset[category]['summaries'][article_id])
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        
        # Tokenize the inputs
        article_tokens = self.tokenizer(
            article,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        summary_tokens = self.tokenizer(
            summary,
            max_length=self.max_length // 4,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': article_tokens['input_ids'].squeeze(),
            'attention_mask': article_tokens['attention_mask'].squeeze(),
            'labels': summary_tokens['input_ids'].squeeze(),
            'summary_attention_mask': summary_tokens['attention_mask'].squeeze()
        }