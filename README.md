# Telecom Support Chatbot with GraphRAG and Ollama

A customer support chatbot for a telecommunications company that uses Graph Retrieval Augmented Generation (GraphRAG) with Ollama (a free, local LLM) to understand customer journeys and provide better issue resolution. No API keys required!

## Project Overview

This chatbot leverages past support tickets, technical manuals, and escalation records to build a knowledge graph that connects:
- Customer issues (e.g., "dropped calls")
- Potential causes (e.g., "network tower outage")
- Effective resolutions (e.g., "reset router")

When a customer asks a question like "My calls keep dropping in the afternoon," the GraphRAG system:
1. Identifies the issue ("dropped calls")
2. Traverses the knowledge graph to find related causes and resolutions
3. Pulls procedural fixes for this specific pattern
4. Delivers a context-rich troubleshooting flow through the LLM

## Features

- **100% Free & Local**: Uses Ollama for local model inference without requiring any API keys
- **GraphRAG Architecture**: Connects issues, causes, and resolutions in a knowledge graph
- **Interactive UI**: Modern Streamlit interface with debug mode and sample queries
- **Ollama Status Indicator**: Shows if Ollama is installed, running, and has the required model
- **Low Memory Footprint**: Configured to use tinyllama model for systems with limited memory

## Project Structure

```
telecom_support_chatbot/
├── data/                      # Data sources
│   ├── support_tickets.csv    # Historical customer support tickets
│   ├── technical_manuals.csv  # Technical documentation and procedures
│   └── escalation_records.csv # Records of escalated support cases
├── src/                       # Source code
│   ├── __init__.py            # Package initialization
│   ├── benchmark.py           # Performance benchmarking tools
│   ├── chatbot.py             # Core chatbot functionality
│   ├── data_exporter.py       # Export functionality for data and results
│   ├── data_processor.py      # Data loading and preprocessing
│   ├── evaluator.py           # Evaluation metrics for chatbot performance
│   ├── graph_rag.py           # Knowledge graph construction and querying
│   ├── local_model.py         # Ollama integration for local LLM inference
│   ├── main.py                # Command-line interface
│   ├── simple_ollama_app.py   # Streamlit web interface optimized for tinyllama
│   ├── test_chatbot.py        # Test cases for the chatbot
│   └── utils.py               # Utility functions
├── docs/                      # Documentation
│   └── knowledge_graph.png    # Visualization of the knowledge graph
├── .env                       # Environment configuration (using tinyllama)
└── requirements.txt           # Project dependencies
```

## Prerequisites

1. **Ollama**: This chatbot requires Ollama for local model inference. Install it from [ollama.ai/download](https://ollama.ai/download)
2. **Python 3.8+**: The chatbot is built with Python 3.8 or higher
3. **TinyLlama Model**: For systems with limited memory (pull with `ollama pull tinyllama`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FathimaHusna/TelecomCustomerSupportGRag.git
cd TelecomCustomerSupportGRag
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Install and start Ollama:
   - Visit [ollama.ai/download](https://ollama.ai/download) and follow the installation instructions for your OS
   - Start the Ollama service
   - Pull the tinyllama model: `ollama pull tinyllama`

## Usage

1. Activate the virtual environment and run the Streamlit app:
```bash
source venv/bin/activate
cd src
streamlit run simple_ollama_app.py
```

2. Open your browser at http://localhost:8501 to interact with the chatbot

3. In the Streamlit interface:
   - Check the Ollama status indicator (green means ready)
   - Click "Initialize Chatbot" in the sidebar
   - Try the sample queries or type your own telecom support questions
   - Toggle debug mode to see additional information

## Screenshot

![Telecom Support Chatbot Screenshot](docs/chatbot_screenshot.png)

The chatbot provides a modern interface with sample queries, debug mode, and real-time responses to telecom support questions.

## Configuration

The `.env` file contains configuration settings:

```
# Ollama Local Model Configuration
LOCAL_MODEL_NAME=tinyllama
OLLAMA_BASE_URL=http://localhost:11434

# Configuration settings
DEBUG=True
LOG_LEVEL=INFO

# Enable GraphRAG visualization
ENABLE_VISUALIZATION=True
```

You can modify these settings to change the model or enable/disable features. The current configuration uses tinyllama, which has a smaller memory footprint suitable for systems with limited RAM. You can also change to any other Ollama-supported model like `llama2:13b`, `mistral`, or `gemma` by updating the `LOCAL_MODEL_NAME` value and pulling that model.

## Advanced Usage

### Debug Mode

Enable debug mode in the Streamlit interface to see:
- Detailed graph query results
- Identified entities and their relationships
- Potential causes and resolutions
- Response generation time

### Export Options

In debug mode, you can export:
- Knowledge graph data (JSON format)
- Conversation history
- Performance metrics

## Troubleshooting

### Ollama Issues

- **Ollama Not Running**: Ensure the Ollama service is running with `ollama serve`
- **Model Not Found**: Pull the required model with `ollama pull llama2`
- **Slow Responses**: Try a smaller model like `llama2` instead of `llama2:13b`

### Application Issues

- **Knowledge Graph Visualization**: If the graph doesn't appear, check that matplotlib is installed
- **Streamlit Interface**: If the UI doesn't load, ensure streamlit is installed with `pip install streamlit`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
- **Entity Extraction**: Identifies key entities in customer queries (issues, devices, network types)
- **Vector Search**: Finds similar past support tickets and relevant documentation
- **Multi-hop Reasoning**: Traverses the knowledge graph to find resolution paths
- **Contextual Responses**: Provides detailed, step-by-step troubleshooting guidance
- **100% Free Local Inference**: Uses Ollama for completely free local inference without API keys
- **Interactive UI**: Clean, modern Streamlit interface with debugging capabilities

## Example Queries

- "My calls keep dropping in the afternoon."
- "Why is my internet so slow during the evening?"
- "I'm being charged for services I didn't subscribe to."
- "I have no service in my basement."
- "My data limit is reached too quickly each month."

## Extending the Project

- **Add More Data**: Expand the dataset with more support tickets and technical documentation
- **Improve Entity Extraction**: Implement more sophisticated NER models
- **Enhance Graph Structure**: Add more relationship types and node attributes
- **Integrate with Support Systems**: Connect to ticketing systems and CRM platforms
- **Add Feedback Loop**: Incorporate agent and customer feedback to improve responses

## License

MIT

## Acknowledgments

This project uses sample data that simulates real-world telecom support scenarios. In a production environment, it would be connected to actual support ticket databases and technical documentation systems.
