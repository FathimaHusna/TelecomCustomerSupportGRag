import os
import json
import csv
import pandas as pd
import networkx as nx
from datetime import datetime

class DataExporter:
    """
    Utility class for exporting data from the telecom support chatbot
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the data exporter
        
        Args:
            output_dir (str): Directory to save exports (defaults to ../exports)
        """
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(os.path.dirname(script_dir), "exports")
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_knowledge_graph(self, graph, format="graphml"):
        """
        Export the knowledge graph to a file
        
        Args:
            graph (nx.Graph): NetworkX graph to export
            format (str): Format to export (graphml, gexf, json)
            
        Returns:
            str: Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "graphml":
            filename = f"knowledge_graph_{timestamp}.graphml"
            filepath = os.path.join(self.output_dir, filename)
            nx.write_graphml(graph, filepath)
        elif format == "gexf":
            filename = f"knowledge_graph_{timestamp}.gexf"
            filepath = os.path.join(self.output_dir, filename)
            nx.write_gexf(graph, filepath)
        elif format == "json":
            filename = f"knowledge_graph_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert graph to dict for JSON serialization
            data = nx.node_link_data(graph)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Knowledge graph exported to {filepath}")
        return filepath
    
    def export_conversation_history(self, conversation_history):
        """
        Export conversation history to a CSV file
        
        Args:
            conversation_history (list): List of conversation messages
            
        Returns:
            str: Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Role', 'Content'])
            
            for message in conversation_history:
                writer.writerow([message.get('role', ''), message.get('content', '')])
        
        print(f"Conversation history exported to {filepath}")
        return filepath
    
    def export_query_results(self, query_results):
        """
        Export query results to a JSON file
        
        Args:
            query_results (dict): Results from a graph query
            
        Returns:
            str: Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Clean up the results for JSON serialization
        clean_results = {
            'query': query_results.get('query', ''),
            'entities': query_results.get('entities', {}),
            'causes': query_results.get('causes', []),
            'resolutions': query_results.get('resolutions', []),
            'paths': [list(path) for path in query_results.get('paths', [])],
            'similar_docs': [
                {
                    'text': doc[0],
                    'metadata': doc[1],
                    'score': float(doc[2])
                }
                for doc in query_results.get('similar_docs', [])
            ],
            'relevant_manuals': query_results.get('relevant_manuals', [])
        }
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Query results exported to {filepath}")
        return filepath
    
    def export_performance_metrics(self, metrics):
        """
        Export performance metrics to a CSV file
        
        Args:
            metrics (dict): Dictionary of performance metrics
            
        Returns:
            str: Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_metrics_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert metrics to DataFrame
        df = pd.DataFrame([metrics])
        df.to_csv(filepath, index=False)
        
        print(f"Performance metrics exported to {filepath}")
        return filepath
