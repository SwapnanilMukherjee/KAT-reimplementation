import pickle
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader
from index import Indexer
import os 
import json
import clip

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]

def encode_sentences(sentences, batch_size=256, device='cuda'):
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = SentenceDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize array to store embeddings
    embeddings = []
    
    # Encode sentences in batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding sentences"):
            # Tokenize text using CLIP's tokenizer
            text = clip.tokenize(batch, truncate=True).to(device)
            
            # Get text features
            text_features = model.encode_text(text)
            
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Move embeddings to CPU and convert to numpy
            embeddings.append(text_features.cpu().numpy())
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def create_faiss_index(embeddings, sentences, entity_ids, wikidata, save_dir="./"):
    """
    Create FAISS index and save all necessary files
    
    Args:
        embeddings: numpy array of embeddings
        sentences: list of sentence strings
        entity_ids: list of entity IDs
        wikidata: original wikidata dictionary
        save_dir: directory to save files
    """
    vector_size = embeddings.shape[1]
    
    # Create indexer
    indexer = Indexer(vector_size)
    
    # Create sequential IDs
    ids = np.arange(len(embeddings))
    
    # Index the embeddings
    indexer.index_data(ids, embeddings)
    
    # Save entity IDs list (maps FAISS index positions to entity IDs)
    entity_path = os.path.join(save_dir, 'entity_ids.pkl')
    with open(entity_path, 'wb') as f:
        pickle.dump(entity_ids, f)
    
    # Save entity data (maps entity IDs to their info)
    entity_data_path = os.path.join(save_dir, 'entity_data.pkl')
    with open(entity_data_path, 'wb') as f:
        pickle.dump(wikidata, f)
    
    # Save index
    indexer.serialize(save_dir)
    
    return indexer

if __name__ == "__main__":
    # Load wikidata
    with open('wikidata_ontology.pkl', 'rb') as f:
        wikidata = pickle.load(f)
    
    # Prepare facts and keep track of entity IDs
    entries = list(wikidata.items())
    entity_ids = []  # to store entity IDs in order
    facts = []       # to store facts in same order
    
    for entity_id, (entity_name, description) in entries:
        fact = f"{entity_name}: {description}"
        facts.append(fact)
        entity_ids.append(entity_id)
    
    print(f"Total facts to encode: {len(facts)}")
    
    # Encode sentences
    embeddings = encode_sentences(
        facts,
        batch_size=1024,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Final embeddings shape: {embeddings.shape}")
    
    # Create faiss index and save all necessary files
    indexer = create_faiss_index(
        embeddings,
        facts,
        entity_ids,
        wikidata,
        save_dir='./'
    )
    print("Faiss index and all necessary files created.")
    
    # Verify the files exist
    required_files = ['index.faiss', 'index_meta.dpr', 'entity_ids.pkl', 'entity_data.pkl']
    for file in required_files:
        path = os.path.join('./', file)
        if os.path.exists(path):
            print(f"Created {file}")
        else:
            print(f"Warning: {file} not found!")