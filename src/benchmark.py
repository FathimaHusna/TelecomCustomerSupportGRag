import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from chatbot import TelecomSupportChatbot
from evaluator import ChatbotEvaluator

class ChatbotBenchmark:
    """
    Benchmark the telecom support chatbot performance
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the benchmark
        
        Args:
            data_dir (str): Path to data directory
        """
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(os.path.dirname(script_dir), "data")
        else:
            self.data_dir = data_dir
            
        self.chatbot = None
        self.evaluator = ChatbotEvaluator()
        self.benchmark_data = None
        self.results = []
    
    def initialize(self):
        """Initialize the chatbot for benchmarking"""
        print("Initializing chatbot for benchmarking...")
        self.chatbot = TelecomSupportChatbot(self.data_dir)
        success = self.chatbot.initialize()
        
        if not success:
            raise Exception("Failed to initialize chatbot")
        
        print("Chatbot initialized successfully")
        return success
    
    def load_benchmark_data(self, filepath=None):
        """
        Load benchmark data from a file
        
        Args:
            filepath (str): Path to benchmark data file
        """
        if filepath is None:
            # Use default benchmark data
            self.benchmark_data = [
                {
                    "query": "My calls keep dropping in the afternoon.",
                    "expected_entities": ["dropped_calls"],
                    "expected_causes": ["Signal interference from nearby construction site", 
                                       "Outdated handoff protocol in specific region"]
                },
                {
                    "query": "Why is my internet so slow during the evening?",
                    "expected_entities": ["slow_internet"],
                    "expected_causes": ["Insufficient bandwidth allocation for growing area", 
                                       "Bandwidth throttling incorrectly applied to account"]
                },
                {
                    "query": "I'm being charged for services I didn't subscribe to.",
                    "expected_entities": ["billing_issue"],
                    "expected_causes": []
                },
                {
                    "query": "I have no service in my basement.",
                    "expected_entities": ["signal_strength"],
                    "expected_causes": ["Building materials blocking signal"]
                },
                {
                    "query": "My data limit is reached too quickly each month.",
                    "expected_entities": ["data_limit"],
                    "expected_causes": []
                }
            ]
            print(f"Loaded {len(self.benchmark_data)} default benchmark queries")
        else:
            # Load from file
            with open(filepath, 'r') as f:
                self.benchmark_data = json.load(f)
            print(f"Loaded {len(self.benchmark_data)} benchmark queries from {filepath}")
    
    def run_benchmark(self):
        """
        Run the benchmark on all test cases
        
        Returns:
            list: Benchmark results
        """
        if not self.chatbot:
            raise Exception("Chatbot not initialized. Call initialize() first.")
            
        if not self.benchmark_data:
            raise Exception("No benchmark data. Call load_benchmark_data() first.")
        
        print(f"Running benchmark on {len(self.benchmark_data)} test cases...")
        self.results = []
        
        for i, test_case in enumerate(tqdm(self.benchmark_data)):
            result = self.run_test_case(test_case, i)
            self.results.append(result)
        
        print("Benchmark completed")
        return self.results
    
    def run_test_case(self, test_case, case_index):
        """
        Run a single test case
        
        Args:
            test_case (dict): Test case data
            case_index (int): Index of the test case
            
        Returns:
            dict: Test case results
        """
        query = test_case["query"]
        expected_entities = test_case.get("expected_entities", [])
        expected_causes = test_case.get("expected_causes", [])
        
        # Measure response time
        start_time = time.time()
        response = self.chatbot.generate_response(query)
        end_time = time.time()
        response_time = end_time - start_time
        
        # Get the latest graph query results
        graph_results = self.chatbot.graph_rag.query_graph(query)
        
        # Evaluate response
        relevance_score = self.evaluator.evaluate_response_relevance(query, response)
        
        # Prepare context for context utilization evaluation
        context = self.chatbot._prepare_context(graph_results)
        context_score = self.evaluator.evaluate_context_utilization(context, response)
        
        # Evaluate path validity
        path_score = self.evaluator.evaluate_path_validity(
            graph_results.get('paths', []), 
            self.chatbot.graph_rag.graph
        )
        
        # Calculate entity match score
        found_entities = set()
        for entity_type, entities in graph_results.get('entities', {}).items():
            found_entities.update(entities)
        
        entity_matches = sum(1 for e in expected_entities if any(e in fe for fe in found_entities))
        entity_score = entity_matches / max(1, len(expected_entities)) if expected_entities else 1.0
        
        # Calculate cause match score
        found_causes = set(graph_results.get('causes', []))
        cause_matches = sum(1 for c in expected_causes if any(c in fc for fc in found_causes))
        cause_score = cause_matches / max(1, len(expected_causes)) if expected_causes else 1.0
        
        # Compile results
        result = {
            "case_index": case_index,
            "query": query,
            "response": response,
            "response_time": response_time,
            "relevance_score": relevance_score,
            "context_score": context_score,
            "path_score": path_score,
            "entity_score": entity_score,
            "cause_score": cause_score,
            "found_entities": list(found_entities),
            "expected_entities": expected_entities,
            "found_causes": list(found_causes),
            "expected_causes": expected_causes,
            "overall_score": (relevance_score + context_score + path_score + entity_score + cause_score) / 5
        }
        
        return result
    
    def save_results(self, output_dir=None):
        """
        Save benchmark results to a file
        
        Args:
            output_dir (str): Directory to save results
            
        Returns:
            str: Path to the saved file
        """
        if not self.results:
            raise Exception("No benchmark results to save")
            
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), "evaluations")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results to JSON
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary to CSV
        summary = []
        for result in self.results:
            summary.append({
                "query": result["query"],
                "response_time": result["response_time"],
                "relevance_score": result["relevance_score"],
                "context_score": result["context_score"],
                "path_score": result["path_score"],
                "entity_score": result["entity_score"],
                "cause_score": result["cause_score"],
                "overall_score": result["overall_score"]
            })
        
        df = pd.DataFrame(summary)
        csv_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Calculate and save aggregate metrics
        aggregates = {
            "total_cases": len(self.results),
            "avg_response_time": df["response_time"].mean(),
            "avg_relevance_score": df["relevance_score"].mean(),
            "avg_context_score": df["context_score"].mean(),
            "avg_path_score": df["path_score"].mean(),
            "avg_entity_score": df["entity_score"].mean(),
            "avg_cause_score": df["cause_score"].mean(),
            "avg_overall_score": df["overall_score"].mean()
        }
        
        agg_path = os.path.join(output_dir, f"benchmark_aggregates_{timestamp}.json")
        with open(agg_path, 'w') as f:
            json.dump(aggregates, f, indent=2)
        
        print(f"Benchmark results saved to {output_dir}")
        print(f"Aggregate score: {aggregates['avg_overall_score']:.4f}")
        
        return json_path

def main():
    parser = argparse.ArgumentParser(description='Benchmark the telecom support chatbot')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--benchmark-file', type=str, help='Path to benchmark data file')
    parser.add_argument('--output-dir', type=str, help='Directory to save results')
    args = parser.parse_args()
    
    benchmark = ChatbotBenchmark(args.data_dir)
    benchmark.initialize()
    benchmark.load_benchmark_data(args.benchmark_file)
    benchmark.run_benchmark()
    benchmark.save_results(args.output_dir)

if __name__ == "__main__":
    main()
