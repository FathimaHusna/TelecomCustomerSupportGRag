import os
import sys
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path

from local_model import LocalModelHandler
from graph_rag import GraphRAG
from data_processor import DataProcessor

# Load environment variables
load_dotenv()

class TelecomSupportChatbot:
    """
    A chatbot for telecom customer support that uses Graph RAG to provide context-aware responses
    """
    
    def __init__(self, data_dir):
        """
        Initialize the chatbot with the data directory
        
        Args:
            data_dir (str): Path to the directory containing the data files
        """
        self.data_dir = data_dir
        self.graph_rag = None
        self.local_model = LocalModelHandler()
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.enable_visualization = os.getenv("ENABLE_VISUALIZATION", "False").lower() == "true"
        self.strict_format = os.getenv("STRICT_FORMAT", "True").lower() == "true"
        
    def initialize(self):
        """
        Initialize the chatbot by loading the knowledge graph and setting up the LLM
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        return self.reload_data()

    def reload_data(self):
        """
        Reload data from disk and rebuild the knowledge graph and vector index.
        Useful for syncing with new data without restarting the service.
        """
        try:
            print(f"Reloading data from {self.data_dir}...")
            # Initialize data pipeline
            data_processor = DataProcessor(self.data_dir)
            if not data_processor.load_data():
                print("Failed to load data files from data directory")
                return False

            data_processor.preprocess_data()
            data_processor.create_embeddings()

            # Build knowledge graph
            graphrag = GraphRAG(data_processor)
            graphrag.build_knowledge_graph()
            self.graph_rag = graphrag
            
            # Visualize the graph if enabled
            if self.enable_visualization:
                self._visualize_graph()
                
            print("Chatbot data successfully reloaded.")
            return True
        except Exception as e:
            print(f"Error during data reload: {e}")
            return False
    
    def _visualize_graph(self):
        """
        Visualize the knowledge graph and save it to a file
        """
        try:
            # Create the docs directory if it doesn't exist
            docs_dir = os.path.join(os.path.dirname(self.data_dir), "docs")
            os.makedirs(docs_dir, exist_ok=True)
            
            # Get the graph from GraphRAG
            G = self.graph_rag.graph
            
            # Create a figure and axis
            plt.figure(figsize=(12, 8))
            
            # Create a layout for the nodes
            pos = nx.spring_layout(G, seed=42)
            
            # Draw the nodes
            node_colors = []
            for node in G.nodes():
                if G.nodes[node].get('type') == 'issue':
                    node_colors.append('lightblue')
                elif G.nodes[node].get('type') == 'cause':
                    node_colors.append('lightcoral')
                elif G.nodes[node].get('type') == 'resolution':
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('gray')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
            
            # Draw the edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            
            # Draw the labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            
            # Save the figure
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(docs_dir, "knowledge_graph.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Knowledge graph visualization saved to docs/knowledge_graph.png")
        except Exception as e:
            print(f"Error visualizing graph: {e}")
    
    def generate_response(self, query):
        """
        Generate a response to the user's query
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The chatbot's response
        """
        try:
            # Query the graph to get relevant context
            graph_results = self.graph_rag.query_graph(query)
            
            # If strict formatting is enabled, deterministically generate a concise answer
            if self.strict_format:
                return self._build_structured_answer(query, graph_results)

            # Otherwise construct the prompt and call the local model
            prompt = self._construct_prompt(query, graph_results)
            response = self.local_model.generate_response(prompt)
            
            # If the model did not follow the expected format, fall back to structured answer
            if not isinstance(response, str) or "Likely Cause:" not in response:
                return self._build_structured_answer(query, graph_results)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def prepare_context(self, graph_results):
        """
        Build the structured context block from graph results
        
        Args:
            graph_results (dict): Results from querying the graph
            
        Returns:
            str: The constructed context section
        """
        context = """
        
# Relevant Information from Knowledge Graph:
"""
        
        # Add information about identified issues (keep the first, most relevant)
        issues = graph_results.get('entities', {}).get('issue_types', []) or []
        if issues:
            primary_issue = issues[0]
            context += f"Identified issue: {primary_issue}\n\n"
        
        # Add potential causes
        if graph_results['causes']:
            context += "Potential causes:\n"
            for cause in list(dict.fromkeys(graph_results['causes']))[:3]:
                context += f"- {cause}\n"
            context += "\n"
        
        # Add recommended resolutions
        if graph_results['resolutions']:
            context += "Recommended resolutions:\n"
            for resolution in list(dict.fromkeys(graph_results['resolutions']))[:3]:
                context += f"- {resolution}\n"
            context += "\n"
        
        # Add relevant paths (issue -> cause -> resolution)
        if graph_results['paths']:
            context += "Relevant troubleshooting paths:\n"
            for path in graph_results['paths'][:2]:
                if len(path) >= 3:
                    context += f"- Issue: {path[0]} → Cause: {path[1]} → Resolution: {path[2]}\n"
                else:
                    context += f"- Path: {' → '.join(path)}\n"
            context += "\n"
        
        # Add information from technical manuals if available
        if 'relevant_manuals' in graph_results and graph_results['relevant_manuals']:
            context += "Relevant technical information:\n"
            for manual in graph_results['relevant_manuals'][:1]:
                context += f"- {manual['title']}: {manual['content'][:200]}...\n"
            context += "\n"
        
        # Add similar support tickets if available
        if 'similar_docs' in graph_results and graph_results['similar_docs']:
            context += "Similar support tickets:\n"
            # Prefer tickets matching the primary issue
            filtered = []
            for i, (doc, metadata, score) in enumerate(graph_results['similar_docs']):
                if metadata.get('source') != 'support_ticket':
                    continue
                if issues and metadata.get('issue_type') not in issues:
                    continue
                filtered.append((i, metadata))
                if len(filtered) >= 1:
                    break
            # Fallback: take the first ticket if no match by issue
            if not filtered:
                for i, (doc, metadata, score) in enumerate(graph_results['similar_docs']):
                    if metadata.get('source') == 'support_ticket':
                        filtered.append((i, metadata))
                        break
            for i, metadata in filtered:
                context += f"- Ticket {metadata.get('id', i)}: {metadata.get('issue_type', 'Unknown')} - {metadata.get('resolution', 'No resolution recorded')}\n"
            context += "\n"
        
        # Add instructions for the model
        context += """
Answer concisely using this format (no greetings or sign-offs):
- Likely Cause: <one sentence based on graph paths/causes>
- Steps: <up to 5 short bullet points with concrete actions>
- Escalate: <one line with clear trigger to contact a human>

Constraints:
- You are assisting a customer, not an engineer.
- Never instruct carrier-only actions (e.g., update cell towers, coverage maps, dispatch technicians, increase capacity).
- Do not suggest router/Wi‑Fi steps unless Wi‑Fi is part of the user’s context.
- Ground only in provided graph/manuals/tickets; do not invent systems or logs.
- Prefer device/network-specific steps when available.
- Keep total answer under 150 words.
"""
        return context

    def _construct_prompt(self, query, graph_results):
        """
        Construct a full prompt for the LLM using the graph results and user query
        
        Args:
            query (str): The user's query
            graph_results (dict): Results from querying the graph
            
        Returns:
            str: The constructed prompt
        """
        prompt_header = f"""You are a telecom customer support assistant. Use the following information to provide a helpful, accurate, and concise response to the customer's query. 

Customer Query: {query}

"""
        context = self.prepare_context(graph_results)
        return prompt_header + context

    def _build_structured_answer(self, query, graph_results):
        """
        Build a concise, grounded response from graph results without relying on generative formatting.
        """
        issues = graph_results.get('entities', {}).get('issue_types', []) or []
        causes = list(dict.fromkeys(graph_results.get('causes', []) or []))
        resolutions = list(dict.fromkeys(graph_results.get('resolutions', []) or []))
        manuals = graph_results.get('relevant_manuals', []) or []
        similar = graph_results.get('similar_docs', []) or []
        devices = graph_results.get('entities', {}).get('devices', []) or []
        networks = graph_results.get('entities', {}).get('network_types', []) or []

        primary_issue = issues[0] if issues else None
        # Issue-specific default likely cause fallbacks for clearer explanations
        issue_defaults = {
            'billing issue': 'Expired/invalid payment method or subscription misconfiguration',
            'billing_issue': 'Expired/invalid payment method or subscription misconfiguration',
            'data limit': 'Background apps consuming data unexpectedly',
            'data_limit': 'Background apps consuming data unexpectedly',
            'dropped call': 'Handoff failure or local signal interference',
            'dropped_calls': 'Handoff failure or local signal interference',
            'signal strength': 'Building materials or location-induced signal shadow',
            'signal_strength': 'Building materials or location-induced signal shadow',
            'voicemail': 'Voicemail not provisioned or device app settings issue',
        }
        likely_cause = None
        if causes:
            likely_cause = causes[0]
        elif primary_issue and primary_issue in issue_defaults:
            likely_cause = issue_defaults[primary_issue]
        elif manuals:
            likely_cause = manuals[0]['title']
        else:
            likely_cause = 'Not identified'
        # Prefer scenario-aware generic cause for cellular dropped calls
        ql = (query or '').lower()
        if primary_issue in ('dropped_calls', 'dropped call') and (any(n in networks for n in ['4g','5g','lte','cellular']) or 'driving' in ql or 'moving' in ql):
            likely_cause = issue_defaults['dropped_calls']
        # Outage real-time question: do not assert past maintenance as current status
        if primary_issue in ('network_outage',) and any(w in ql for w in ['today', 'now', 'right now']):
            likely_cause = 'Cannot confirm real-time status without live network data'

        # Filter to user-actionable resolutions based on context
        def _filter_user_actionable(items):
            disallow_ops = [
                'cell tower', 'coverage map', 'tower maintenance', 'microcell',
                'repaired damaged', 'storm', 'fiber optic', 'updated cell tower handoff',
                'rerouted traffic', 'bandwidth allocation', 'increase capacity', 'dispatch', 'field technician'
            ]
            out = []
            for s in items:
                s_low = str(s).lower()
                if ('wifi' not in networks) and ('router' in s_low):
                    continue
                if any(k in s_low for k in disallow_ops):
                    continue
                out.append(s)
            return out

        # Build steps: prefer explicit, user-actionable resolutions
        steps = []
        for r in _filter_user_actionable(resolutions)[:3]:
            steps.append(r)
        # If we still lack concrete steps, try similar support tickets for matching issue
        if not steps:
            # Prefer tickets whose issue_type matches the primary issue
            for doc, metadata, score in similar:
                if metadata.get('source') == 'support_ticket':
                    if (not primary_issue) or (metadata.get('issue_type') == primary_issue):
                        res = metadata.get('resolution')
                        if res:
                            filtered = _filter_user_actionable([res])
                            for r in filtered:
                                if r not in steps:
                                    steps.append(r)
                        if len(steps) >= 3:
                            break
        # Deterministic fallback: pull resolutions from historical tickets by issue type
        if not steps and primary_issue and getattr(self, 'graph_rag', None):
            try:
                dp = getattr(self.graph_rag, 'data_processor', None)
                if dp is not None and getattr(dp, 'support_tickets', None) is not None:
                    # Map entity issue labels to ticket issue_type values
                    issue_map = {
                        'billing issue': 'billing_issue',
                        'dropped call': 'dropped_calls',
                        'slow internet': 'slow_internet',
                        'signal strength': 'signal_strength',
                        'no service': 'network_outage',
                    }
                    canonical_issue = issue_map.get(primary_issue, primary_issue)
                    try:
                        import pandas as _pd  # local import to avoid top-level dependency here
                        mask = dp.support_tickets['issue_type'] == canonical_issue
                        matched = dp.support_tickets[mask]
                        if not matched.empty:
                            ticket_resolutions = matched['resolution'].dropna().tolist()
                            for res in ticket_resolutions:
                                if res:
                                    filtered = _filter_user_actionable([res])
                                    for r in filtered:
                                        if r not in steps:
                                            steps.append(r)
                                            if len(steps) >= 3:
                                                break
                    except Exception:
                        pass
            except Exception:
                pass
        # Contextual, device/network-specific generic steps for common issues
        t = (query or '').lower()
        # Add contextual steps for common issues (append even if some steps exist)
        if primary_issue in ('dropped_calls', 'dropped call'):
            if any(n in networks for n in ['4g', '5g', 'lte', 'cellular']) or ('driving' in t or 'moving' in t):
                cellular_steps = [
                    'Toggle Airplane Mode for 10 seconds, then turn off',
                    'Check for carrier settings update (Settings > General > About)',
                    'Reseat SIM: power off, remove, clean, reinsert',
                    'Reset Network Settings (Settings > General > Transfer or Reset > Reset > Reset Network Settings)'
                ]
                for s in cellular_steps:
                    if s not in steps:
                        steps.append(s)
                if devices:
                    hint = f"Update OS and carrier settings on {devices[0]}"
                    if hint not in steps:
                        steps.append(hint)
            elif 'wifi' in networks or 'wifi' in t:
                wifi_calling_steps = [
                    'Enable Wi‑Fi Calling in device settings',
                    'Check router placement and reduce interference'
                ]
                for s in wifi_calling_steps:
                    if s not in steps:
                        steps.append(s)
        if primary_issue in ('slow_internet', 'slow internet'):
            if 'wifi' in networks or 'wifi' in t:
                wifi_steps = [
                    'Move closer to router; prefer 5 GHz if available',
                    'Reduce interference (microwave, thick walls); try a different Wi‑Fi channel',
                    'Restart router and modem; check for firmware updates'
                ]
                for s in wifi_steps:
                    if s not in steps:
                        steps.append(s)
            else:
                cell_steps = [
                    'Test at non‑peak hours to rule out congestion',
                    'Disable background data and auto‑updates temporarily',
                    'Toggle Airplane Mode and retry a speed test'
                ]
                for s in cell_steps:
                    if s not in steps:
                        steps.append(s)
        if primary_issue in ('voicemail',):
            if 'android' in devices:
                vm_android = [
                    'Force stop Phone app and clear cache',
                    'Check call forwarding is off',
                    'Call voicemail, reset PIN, and re‑setup'
                ]
                for s in vm_android:
                    if s not in steps:
                        steps.append(s)
            else:
                vm_ios = [
                    'Call voicemail to re‑setup; verify PIN',
                    'Check carrier settings update (Settings > General > About)',
                    'Reset Network Settings if issue persists'
                ]
                for s in vm_ios:
                    if s not in steps:
                        steps.append(s)
        if primary_issue in ('network_outage','no service'):
            outage_steps = [
                'Check provider status page/alerts for your area',
                'Toggle Airplane Mode and retry from an open area',
                'Ask a nearby user on same carrier to confirm'
            ]
            for s in outage_steps:
                if s not in steps:
                    steps.append(s)
        if not steps and manuals:
            # Extract 1-2 actionable sentences from the manual content
            snippet = manuals[0]['content'][:200]
            steps.append(f"Follow guidance in '{manuals[0]['title']}' (see admin/guide)")
        # Add a device/network hint if detected
        if devices:
            steps.append(f"Check device-specific settings on {devices[0]}")
        if networks:
            steps.append(f"Test on {networks[0]} and compare with alternative network")
        # Rewrite past-tense/upgrade steps into customer-friendly actions
        def _rewrite_step(s):
            sl = s.lower()
            # Past-tense engineer diary → customer instructions
            if 'identified' in sl and 'replac' in sl and 'sim' in sl:
                return 'Reseat SIM: power off, remove, clean, reinsert; if still failing, visit a store to replace the SIM'
            # Generic replace/upgrade as last resort in Wi‑Fi context
            if 'upgrade' in sl and 'router' in sl:
                return 'Use 5 GHz band and reduce interference; if the router is old and issues persist, consider a dual‑band upgrade'
            if 'replac' in sl and 'router' in sl:
                return 'Restart router and check firmware; if hardware is faulty and issues persist, consider replacement'
            return s
        steps = [
            _rewrite_step(s) for s in steps
        ]
        # Deduplicate while preserving order and cap to 5
        seen = set()
        deduped = []
        for s in steps:
            if s not in seen:
                deduped.append(s)
                seen.add(s)
        steps = deduped[:5]

        # Escalation rule of thumb
        if primary_issue in ('network_outage', 'signal_strength'):
            escalate = "Escalate if area-wide or persists after steps; note location/time."
        elif primary_issue in ('billing_issue', 'billing issue'):
            escalate = "Escalate if unauthorized charges remain after subscription/payment correction."
        elif primary_issue in ('dropped_calls', 'dropped call'):
            escalate = "Escalate if drops persist across locations or after steps; mention driving route/time."
        else:
            escalate = "Escalate if unresolved after steps or safety-impacting."

        # Compose final capped answer
        def bulletize(items):
            return '\n'.join([f"- {s}" for s in items])

        out = []
        out.append(f"Likely Cause: {likely_cause}")
        if steps:
            out.append("Steps:")
            out.append(bulletize(steps))
        out.append(f"Escalate: {escalate}")
        text = '\n'.join(out)
        # Truncate to ~150 words
        words = text.split()
        if len(words) > 160:
            text = ' '.join(words[:160])
        return text
