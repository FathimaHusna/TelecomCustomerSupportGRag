import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Telecom Support (Graph Only)",
    page_icon="🕸️",
    layout="wide",
)

st.title("Telecom Customer Support — GraphRAG Only")
st.caption("Runs without Ollama or external APIs")

from chatbot import TelecomSupportChatbot

@st.cache_resource(show_spinner=True)
def init_bot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    bot = TelecomSupportChatbot(data_dir)
    ok = bot.initialize()
    return bot, ok

bot, ok = init_bot()
if not ok:
    st.error("Initialization failed. Check data files and logs.")
else:
    st.success("Graph initialized. Ask a question below.")

query = st.text_input("How can I help you today?")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        # Strict format is enabled by default; uses graph/manuals only
        ans = bot.generate_response(query)
        st.markdown(ans)

st.markdown("---")
st.caption("GraphRAG pipeline: tickets + manuals + escalations → knowledge graph → grounded responses")

