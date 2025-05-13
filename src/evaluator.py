import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class ChatbotEvaluator:
    """
    Evaluator for the telecom support chatbot
    Provides metrics and evaluation tools to measure chatbot performance
    """
    
    def __init__(self):
        """Initialize the chatbot evaluator"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.metrics = {
            'response_relevance': [],
            'context_utilization': [],
            'response_time': [],
            'path_validity': []
        }
        
    def evaluate_response_relevance(self, query, response, reference_responses=None):
        """
        Evaluate the relevance of a response to the query
        
        Args:
            query (str): User query
            response (str): Chatbot response
            reference_responses (list): Optional list of reference responses
            
        Returns:
            float: Relevance score between 0 and 1
        """
        # If reference responses are provided, compare to them
        if reference_responses:
            # Embed the response and reference responses
            response_embedding = self.embedding_model.encode([response])[0]
            reference_embeddings = self.embedding_model.encode(reference_responses)
            
            # Calculate cosine similarity with each reference response
            similarities = cosine_similarity([response_embedding], reference_embeddings)[0]
            
            # Return the maximum similarity
            relevance_score = float(np.max(similarities))
        else:
            # Otherwise, compare query and response directly
            query_embedding = self.embedding_model.encode([query])[0]
            response_embedding = self.embedding_model.encode([response])[0]
            
            # Calculate cosine similarity
            relevance_score = float(cosine_similarity([query_embedding], [response_embedding])[0][0])
        
        self.metrics['response_relevance'].append(relevance_score)
        return relevance_score
    
    def evaluate_context_utilization(self, context, response):
        """
        Evaluate how well the response utilizes the provided context
        
        Args:
            context (str): Context provided to the LLM
            response (str): Chatbot response
            
        Returns:
            float: Context utilization score between 0 and 1
        """
        # Extract key phrases from context
        context_lines = context.split('\n')
        key_phrases = []
        
        for line in context_lines:
            if line.strip().startswith('-'):
                # Extract the content after the bullet point
                phrase = line.strip()[1:].strip()
                if len(phrase) > 5:  # Only consider substantial phrases
                    key_phrases.append(phrase)
        
        if not key_phrases:
            return 0.5  # Default score if no key phrases found
        
        # Count how many key phrases are mentioned in the response
        mention_count = 0
        for phrase in key_phrases:
            # Check if any part of the phrase (at least 5 chars) is in the response
            words = phrase.split()
            for i in range(len(words)):
                for j in range(i+1, len(words)+1):
                    sub_phrase = ' '.join(words[i:j])
                    if len(sub_phrase) > 5 and sub_phrase.lower() in response.lower():
                        mention_count += 1
                        break
                if mention_count > 0:
                    break
        
        # Calculate utilization score
        utilization_score = min(1.0, mention_count / max(1, len(key_phrases)))
        
        self.metrics['context_utilization'].append(utilization_score)
        return utilization_score
    
    def evaluate_response_time(self, start_time, end_time):
        """
        Evaluate the response time
        
        Args:
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            
        Returns:
            float: Response time in seconds
        """
        response_time = end_time - start_time
        self.metrics['response_time'].append(response_time)
        return response_time
    
    def evaluate_path_validity(self, paths, graph):
        """
        Evaluate the validity of paths in the knowledge graph
        
        Args:
            paths (list): List of paths returned by the GraphRAG
            graph (nx.Graph): The knowledge graph
            
        Returns:
            float: Path validity score between 0 and 1
        """
        if not paths:
            return 0.0
        
        valid_paths = 0
        for path in paths:
            # Check if the path exists in the graph
            is_valid = True
            for i in range(len(path) - 1):
                if not graph.has_edge(path[i], path[i+1]):
                    is_valid = False
                    break
            
            if is_valid:
                valid_paths += 1
        
        validity_score = valid_paths / len(paths)
        self.metrics['path_validity'].append(validity_score)
        return validity_score
    
    def get_aggregate_metrics(self):
        """
        Get aggregate metrics across all evaluations
        
        Returns:
            dict: Dictionary of aggregate metrics
        """
        aggregates = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                aggregates[f'{metric_name}_mean'] = np.mean(values)
                aggregates[f'{metric_name}_median'] = np.median(values)
                aggregates[f'{metric_name}_min'] = np.min(values)
                aggregates[f'{metric_name}_max'] = np.max(values)
                aggregates[f'{metric_name}_count'] = len(values)
        
        return aggregates
    
    def save_evaluation_results(self, output_dir=None):
        """
        Save evaluation results to a file
        
        Args:
            output_dir (str): Directory to save results (defaults to ../evaluations)
            
        Returns:
            str: Path to the saved file
        """
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), "evaluations")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get aggregate metrics
        results = self.get_aggregate_metrics()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results['timestamp'] = timestamp
        
        # Save to CSV
        filename = f"evaluation_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame([results])
        df.to_csv(filepath, index=False)
        
        print(f"Evaluation results saved to {filepath}")
        return filepath
