import networkx as nx
import pandas as pd
import numpy as np
from data_processor import DataProcessor
import matplotlib.pyplot as plt
import os

class GraphRAG:
    def __init__(self, data_processor):
        """
        Initialize the GraphRAG system with a data processor
        
        Args:
            data_processor (DataProcessor): Initialized data processor with loaded data
        """
        self.data_processor = data_processor
        self.graph = nx.DiGraph()
        self.issue_to_cause_map = {}
        self.cause_to_resolution_map = {}
        
    @staticmethod
    def _normalize(text):
        try:
            return str(text).lower().replace('_', ' ').strip()
        except Exception:
            return str(text).lower()
        
    def build_knowledge_graph(self):
        """Build the knowledge graph from the processed data"""
        print("Building knowledge graph...")
        
        # Add nodes for issue types
        issue_types = self.data_processor.support_tickets['issue_type'].unique()
        for issue in issue_types:
            self.graph.add_node(issue, type='issue')
            
        # Add nodes for devices
        device_types = self.data_processor.support_tickets['device_type'].dropna().unique()
        for device in device_types:
            self.graph.add_node(device, type='device')
            
        # Add nodes for network types
        network_types = self.data_processor.support_tickets['network_type'].dropna().unique()
        for network in network_types:
            self.graph.add_node(network, type='network')
            
        # Add root causes as nodes from escalation records
        for _, row in self.data_processor.escalation_records.iterrows():
            cause = row['root_cause']
            self.graph.add_node(cause, type='cause')
            
            # Find the associated ticket
            ticket_id = row['ticket_id']
            ticket_row = self.data_processor.support_tickets[
                self.data_processor.support_tickets['ticket_id'] == ticket_id
            ]
            
            if not ticket_row.empty:
                issue_type = ticket_row['issue_type'].values[0]
                resolution = ticket_row['resolution'].values[0]
                
                # Add resolution as node
                self.graph.add_node(resolution, type='resolution')
                
                # Connect issue to cause
                self.graph.add_edge(issue_type, cause, weight=1.0)
                
                # Connect cause to resolution
                self.graph.add_edge(cause, resolution, weight=1.0)
                
                # Update maps
                if issue_type not in self.issue_to_cause_map:
                    self.issue_to_cause_map[issue_type] = []
                self.issue_to_cause_map[issue_type].append(cause)
                
                if cause not in self.cause_to_resolution_map:
                    self.cause_to_resolution_map[cause] = []
                self.cause_to_resolution_map[cause].append(resolution)
        
        # Connect issues to devices and networks
        for _, row in self.data_processor.support_tickets.iterrows():
            issue = row['issue_type']
            device = row['device_type']
            network = row['network_type']
            
            if pd.notna(device) and device in self.graph:
                self.graph.add_edge(issue, device, weight=0.5)
                
            if pd.notna(network) and network in self.graph:
                self.graph.add_edge(issue, network, weight=0.5)
        
        # Add technical manual knowledge
        for _, row in self.data_processor.technical_manuals.iterrows():
            title = row['title']
            category = row['category']
            device = row['device_type']
            network = row['network_type']
            
            # Add manual as node
            self.graph.add_node(title, type='manual')
            
            # Connect to relevant device and network types
            if device in self.graph:
                self.graph.add_edge(device, title, weight=0.7)
                
            if network in self.graph:
                self.graph.add_edge(network, title, weight=0.7)
                
            # Connect to relevant issues based on keywords
            for issue in issue_types:
                if issue.replace('_', ' ') in title.lower() or issue.replace('_', ' ') in row['content'].lower():
                    self.graph.add_edge(issue, title, weight=0.8)
        
        print(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def visualize_graph(self, output_path=None):
        """
        Visualize the knowledge graph
        
        Args:
            output_path (str): Path to save the visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            if node_type == 'issue':
                node_colors.append('red')
            elif node_type == 'cause':
                node_colors.append('green')
            elif node_type == 'resolution':
                node_colors.append('blue')
            elif node_type == 'device':
                node_colors.append('orange')
            elif node_type == 'network':
                node_colors.append('purple')
            elif node_type == 'manual':
                node_colors.append('brown')
            else:
                node_colors.append('gray')
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, alpha=0.8, node_size=200)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrows=True)
        
        # Add labels to a subset of nodes to avoid clutter
        labels = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') in ['issue', 'cause']:
                # Truncate long labels
                labels[node] = str(node)[:20] + '...' if len(str(node)) > 20 else str(node)
        
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        plt.title("Telecom Support Knowledge Graph")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Graph visualization saved to {output_path}")
        else:
            plt.show()
    
    def query_graph(self, query, k=3):
        """
        Query the knowledge graph using the input query
        
        Args:
            query (str): User query
            k (int): Number of paths to return
            
        Returns:
            dict: Dictionary containing query results
        """
        # Extract entities from query
        entities = self.data_processor.extract_entities(query)
        
        # Find similar documents using vector search
        similar_docs = self.data_processor.search_similar_documents(query, k=5)
        
        # Identify relevant nodes in the graph
        relevant_nodes = set()
        
        # Add issue types from entities (normalized match to handle underscores vs spaces)
        for issue in entities['issue_types']:
            issue_norm = self._normalize(issue)
            for graph_node in self.graph.nodes():
                if issue_norm in self._normalize(graph_node):
                    relevant_nodes.add(graph_node)
        
        # Add devices from entities
        for device in entities['devices']:
            for graph_node in self.graph.nodes():
                if device in str(graph_node).lower() and self.graph.nodes[graph_node].get('type') == 'device':
                    relevant_nodes.add(graph_node)
        
        # Add network types from entities
        for network in entities['network_types']:
            for graph_node in self.graph.nodes():
                if network in str(graph_node).lower() and self.graph.nodes[graph_node].get('type') == 'network':
                    relevant_nodes.add(graph_node)
        
        # If no relevant nodes found from entities, use similar documents
        if not relevant_nodes:
            for doc, metadata, _ in similar_docs:
                if metadata['source'] == 'support_ticket':
                    issue_type = metadata['issue_type']
                    if issue_type in self.graph:
                        relevant_nodes.add(issue_type)
        
        # Find paths in the graph
        paths = []
        causes = []
        resolutions = []
        
        for node in relevant_nodes:
            # If node is an issue, find paths to resolutions
            if self.graph.nodes[node].get('type') == 'issue':
                # Find causes for this issue
                if node in self.issue_to_cause_map:
                    for cause in self.issue_to_cause_map[node]:
                        causes.append(cause)
                        
                        # Find resolutions for this cause
                        if cause in self.cause_to_resolution_map:
                            for resolution in self.cause_to_resolution_map[cause]:
                                resolutions.append(resolution)
                                paths.append((node, cause, resolution))
        
        # Do not invent cross-issue paths; if no explicit mappings, keep paths empty
        
        # Limit to top k paths
        paths = paths[:k]
        
        # Get relevant technical manuals: neighbors first, then fall back to vector hits
        relevant_manuals = []
        manual_titles_added = set()
        for node in relevant_nodes:
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor].get('type') == 'manual':
                    # Find the manual details
                    for _, row in self.data_processor.technical_manuals.iterrows():
                        if row['title'] == neighbor and row['title'] not in manual_titles_added:
                            relevant_manuals.append({
                                'title': row['title'],
                                'content': row['content']
                            })
                            manual_titles_added.add(row['title'])

        # If none found via neighbors, include top technical manuals from vector search
        if not relevant_manuals:
            for doc, metadata, _ in similar_docs:
                if metadata.get('source') == 'technical_manual':
                    title = metadata.get('title')
                    # Look up full text from dataframe
                    match = self.data_processor.technical_manuals[self.data_processor.technical_manuals['title'] == title]
                    if not match.empty and title not in manual_titles_added:
                        relevant_manuals.append({
                            'title': title,
                            'content': match.iloc[0]['content']
                        })
                        manual_titles_added.add(title)

        return {
            'query': query,
            'entities': entities,
            'similar_docs': similar_docs,
            'paths': paths,
            'causes': causes,
            'resolutions': resolutions,
            'relevant_manuals': relevant_manuals
        }
