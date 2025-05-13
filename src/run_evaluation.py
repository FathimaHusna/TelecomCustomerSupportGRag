import os
import sys
import json
import time
import pandas as pd
from tqdm import tqdm
from chatbot import TelecomSupportChatbot
from evaluator import ChatbotEvaluator
from data_processor import DataProcessor
from graph_rag import GraphRAG

def run_evaluation():
    """Run a simple evaluation of the chatbot"""
    print("Starting evaluation of telecom support chatbot...")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    output_dir = os.path.join(os.path.dirname(script_dir), "evaluations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components manually to bypass the initialization issue
    print("Initializing components...")
    data_processor = DataProcessor(data_dir)
    if not data_processor.load_data():
        print("Failed to load data")
        return
    
    data_processor.preprocess_data()
    data_processor.create_embeddings()
    
    graph_rag = GraphRAG(data_processor)
    graph_rag.build_knowledge_graph()
    
    chatbot = TelecomSupportChatbot(data_dir)
    chatbot.graph_rag = graph_rag
    chatbot.local_model = chatbot.local_model  # Already initialized in __init__
    
    evaluator = ChatbotEvaluator()
    
    # Test queries
    test_queries = [
        "My calls keep dropping in the afternoon",
        "Why is my internet so slow during peak hours?",
        "I'm being charged for services I didn't subscribe to",
        "I have no signal in my basement",
        "My data limit is reached too quickly each month"
    ]
    
    results = []
    
    print("\nRunning evaluation on test queries...")
    for query in tqdm(test_queries):
        # Get response
        start_time = time.time()
        response = chatbot.generate_response(query)
        end_time = time.time()
        response_time = end_time - start_time
        
        # Get graph query results
        graph_results = chatbot.graph_rag.query_graph(query)
        
        # Evaluate response
        relevance_score = evaluator.evaluate_response_relevance(query, response)
        
        # Prepare context for context utilization evaluation
        # Create a simple context string from graph results
        context = ""
        if 'paths' in graph_results and graph_results['paths']:
            context += "Paths found in knowledge graph:\n"
            for i, path in enumerate(graph_results['paths']):
                path_str = " -> ".join([str(node) for node in path])
                context += f"- {path_str}\n"
        
        if 'causes' in graph_results and graph_results['causes']:
            context += "\nPotential causes:\n"
            for cause in graph_results['causes']:
                context += f"- {cause}\n"
                
        if 'resolutions' in graph_results and graph_results['resolutions']:
            context += "\nRecommended resolutions:\n"
            for resolution in graph_results['resolutions']:
                context += f"- {resolution}\n"
                
        context_score = evaluator.evaluate_context_utilization(context, response)
        
        # Evaluate path validity
        path_score = evaluator.evaluate_path_validity(
            graph_results.get('paths', []), 
            chatbot.graph_rag.graph
        )
        
        # Calculate overall score
        overall_score = (relevance_score + context_score + path_score) / 3
        
        # Store results
        result = {
            "query": query,
            "response": response,
            "response_time": response_time,
            "relevance_score": relevance_score,
            "context_score": context_score,
            "path_score": path_score,
            "overall_score": overall_score
        }
        results.append(result)
        
        # Print individual result
        print(f"\n\n{'='*80}")
        print(f"Query: {query}")
        print(f"Response: {response[:200]}...")
        print(f"Evaluation Metrics:")
        print(f"  - Response Relevance: {relevance_score:.4f}")
        print(f"  - Context Utilization: {context_score:.4f}")
        print(f"  - Path Validity: {path_score:.4f}")
        print(f"  - Overall Score: {overall_score:.4f}")
        print(f"{'='*80}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate and save aggregate metrics
    df = pd.DataFrame(results)
    aggregates = {
        "total_queries": len(results),
        "avg_response_time": df["response_time"].mean(),
        "avg_relevance_score": df["relevance_score"].mean(),
        "avg_context_score": df["context_score"].mean(),
        "avg_path_score": df["path_score"].mean(),
        "avg_overall_score": df["overall_score"].mean()
    }
    
    agg_path = os.path.join(output_dir, f"evaluation_aggregates_{timestamp}.json")
    with open(agg_path, 'w') as f:
        json.dump(aggregates, f, indent=2)
    
    print("\nEvaluation Summary:")
    print(f"Total queries: {aggregates['total_queries']}")
    print(f"Average response time: {aggregates['avg_response_time']:.2f} seconds")
    print(f"Average relevance score: {aggregates['avg_relevance_score']:.4f}")
    print(f"Average context utilization: {aggregates['avg_context_score']:.4f}")
    print(f"Average path validity: {aggregates['avg_path_score']:.4f}")
    print(f"Average overall score: {aggregates['avg_overall_score']:.4f}")
    
    print(f"\nEvaluation results saved to {output_dir}")
    return json_path

if __name__ == "__main__":
    run_evaluation()
