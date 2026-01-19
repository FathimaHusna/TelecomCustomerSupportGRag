import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class DataProcessor:
    def __init__(self, data_dir):
        """
        Initialize the data processor with the directory containing data files.
        
        Args:
            data_dir (str): Path to the directory containing data files
        """
        self.data_dir = data_dir
        self.support_tickets = None
        self.technical_manuals = None
        self.escalation_records = None
        # Configure embedding device (default to CPU to avoid CUDA issues)
        self.embedding_device = os.getenv('EMBEDDING_DEVICE', 'cpu').lower()
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.embedding_device)
        except Exception as e:
            print(f"Error initializing embedding model on device '{self.embedding_device}': {e}")
            print("Falling back to CPU for embeddings.")
            self.embedding_device = 'cpu'
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.document_embeddings = None
        self.document_texts = []
        self.document_metadata = []
        self.index = None
        
    def load_data(self):
        """Load all data sources from the data directory"""
        try:
            self.support_tickets = pd.read_csv(os.path.join(self.data_dir, 'support_tickets.csv'))
            self.technical_manuals = pd.read_csv(os.path.join(self.data_dir, 'technical_manuals.csv'))
            self.escalation_records = pd.read_csv(os.path.join(self.data_dir, 'escalation_records.csv'))
            print(f"Loaded {len(self.support_tickets)} support tickets")
            print(f"Loaded {len(self.technical_manuals)} technical manuals")
            print(f"Loaded {len(self.escalation_records)} escalation records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess and combine all data sources for embedding"""
        # Process support tickets
        for _, row in self.support_tickets.iterrows():
            text = f"Ticket: {row['issue_type']}. Description: {row['description']}. Resolution: {row['resolution']}"
            self.document_texts.append(text)
            self.document_metadata.append({
                'source': 'support_ticket',
                'id': row['ticket_id'],
                'issue_type': row['issue_type'],
                'resolution': row['resolution'],
                'device_type': row['device_type'],
                'network_type': row['network_type']
            })
        
        # Process technical manuals
        for _, row in self.technical_manuals.iterrows():
            text = f"Manual: {row['title']}. Category: {row['category']}. Content: {row['content']}"
            self.document_texts.append(text)
            self.document_metadata.append({
                'source': 'technical_manual',
                'id': row['manual_id'],
                'title': row['title'],
                'category': row['category'],
                'device_type': row['device_type'],
                'network_type': row['network_type']
            })
        
        # Process escalation records
        for _, row in self.escalation_records.iterrows():
            text = f"Escalation for ticket {row['ticket_id']}. Root cause: {row['root_cause']}. Resolution steps: {row['resolution_steps']}"
            self.document_texts.append(text)
            self.document_metadata.append({
                'source': 'escalation_record',
                'id': row['escalation_id'],
                'ticket_id': row['ticket_id'],
                'root_cause': row['root_cause'],
                'resolution_steps': row['resolution_steps'],
                'escalation_level': row['escalation_level']
            })
        
        print(f"Preprocessed {len(self.document_texts)} total documents")
        return len(self.document_texts)
    
    def create_embeddings(self):
        """Create embeddings for all preprocessed documents"""
        print("Creating embeddings for all documents...")
        try:
            self.document_embeddings = self.embedding_model.encode(
                self.document_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        except Exception as e:
            # Safety fallback in case of CUDA/driver issues during encode
            if self.embedding_device != 'cpu':
                print(f"Error during embedding on device '{self.embedding_device}': {e}")
                print("Retrying on CPU...")
                self.embedding_device = 'cpu'
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                self.document_embeddings = self.embedding_model.encode(
                    self.document_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            else:
                raise
        
        # Create FAISS index for fast similarity search
        dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.document_embeddings).astype('float32'))
        
        print(f"Created embeddings with dimension {dimension}")
        return self.document_embeddings.shape
    
    def search_similar_documents(self, query, k=5):
        """
        Search for documents similar to the query
        
        Args:
            query (str): The query text
            k (int): Number of results to return
            
        Returns:
            list: List of (document_text, metadata, score) tuples
        """
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.document_texts):
                results.append((
                    self.document_texts[idx],
                    self.document_metadata[idx],
                    float(distances[0][i])
                ))
        
        return results
    
    def extract_entities(self, text):
        """
        Extract telecom-specific entities from text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of extracted entities
        """
        entities = {
            'issue_types': [],
            'devices': [],
            'network_types': [],
            'locations': []
        }
        
        # Simple rule-based entity extraction with canonical issue mapping
        # In a production system, this would be replaced with an NER model
        t = text.lower()
        
        # Canonical issue types and their keyword heuristics
        issue_synonyms = {
            'dropped_calls': [r'dropped\s*call', r'call\s*drop', r'calls?\s*keep\s*dropp', r'handoff', r'while\s*driving'],
            'slow_internet': [r'slow\s*internet', r'slow\s*speed', r'peak\s*hours', r'unusable\s*during\s*evening', r'bandwidth'],
            'billing_issue': [r'billing', r'charged', r'charge', r'subscription', r'subscribe', r'refund', r'invoice', r'payment', r'autopay'],
            'network_outage': [r'no\s*service', r'outage', r'downtime', r'tower', r'downtown\s*area'],
            'signal_strength': [r'signal\s*strength', r'weak\s*signal', r'no\s*reception', r'coverage', r'basement'],
            'data_limit': [r'data\s*limit', r'data\s*cap', r'ran\s*out\s*of\s*data', r'data\s*usage', r'background\s*data'],
            'voicemail': [r'voicemail', r'voice\s*mail']
        }
        for canonical, patterns in issue_synonyms.items():
            for pat in patterns:
                if re.search(pat, t):
                    if canonical not in entities['issue_types']:
                        entities['issue_types'].append(canonical)
                    break
        
        # Extract devices
        device_patterns = ['iphone', 'samsung', 'pixel', 'router', 'modem', 'android']
        for pattern in device_patterns:
            if re.search(pattern, t):
                entities['devices'].append(pattern)
        
        # Extract network types
        network_patterns = ['4g', '5g', 'wifi', 'lte', 'cellular']
        for pattern in network_patterns:
            if re.search(pattern, t):
                entities['network_types'].append(pattern)
        
        return entities
