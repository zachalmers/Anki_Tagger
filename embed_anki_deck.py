import os
import openai
import tiktoken
import pandas as pd
from openai.embeddings_utils import get_embedding

def set_api_key():
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    except KeyError:
        #In terminal: export OPENAI_API_KEY=your-api-key
        print("Set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'")

def load_dataset(input_datapath):
    if not os.path.exists(input_datapath):
        print(f"{input_datapath} does not exist. Please check your file path.")
        return None

    df = pd.read_csv(input_datapath, sep='\t', header=None, usecols=[0,1], names=["guid", "card"], comment='#')
    df = df.dropna()
    return df

def filter_by_tokens(df, encoding, max_tokens):
    df["tokens"] = df.card.apply(lambda x: len(encoding.encode(x)))
    df = df[df.tokens <= max_tokens]
    return df

def calculate_embeddings(df, embedding_model):
    df["emb"] = df.card.apply(lambda x: get_embedding(x, engine=embedding_model))
    return df

def save_embeddings(df, output_prefix):
    df.to_csv(f"./{output_prefix}_embeddings.csv", index=False)

if __name__ == "__main__":
    
    # Set OpenAI API key and embedding model parameters
    set_api_key()
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # This is the encoding for text-embedding-ada-002
    max_tokens = 8000  # The maximum for text-embedding-ada-002 is 8191

    # Set deck to embed.
    #This is the deck you'll apply your tags to in the end.
    #In anki, export deck notes as plain text with guid flag checked
    input_datapath = "./anking_notes_plaintxt_with_guid.txt"
    output_prefix = "your_deck_name" # EDIT AS NEEDED

    # Load and preprocess dataset
    df = load_dataset(input_datapath)
    if df is not None:
        encoding = tiktoken.get_encoding(embedding_encoding)
        df = filter_by_tokens(df, encoding, max_tokens)

        # Calculate embeddings for cards
        df = calculate_embeddings(df, embedding_model)

        # Save embeddings to file
        save_embeddings(df, output_prefix)
