import streamlit as st
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pipeline import RAGPipeline

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize session state for pipeline
if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline(
            data_dir="../data",
            persist_dir="chroma_index"
        )
        st.success("✅ Successfully initialized the pipeline!")
    except Exception as e:
        st.error(f"""❌ Error initializing pipeline: {str(e)}

Please make sure you have all required packages installed and environment variables set.
""")
        st.stop()

# Page config
st.set_page_config(
    page_title="INGRES-AI RAG System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        .search-result {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        h1 {
            color: #1E3D59;
        }
        .stMarkdown {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📚 INGRES-AI Document Search")
st.markdown("### Intelligent Document Search and Question Answering System")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar controls
with st.sidebar:
    if st.button("Index Documents"):
        with st.spinner("Indexing documents..."):
            try:
                st.session_state.pipeline.index_documents()
                st.success("✅ Documents indexed successfully!")
            except Exception as e:
                st.error(f"❌ Error indexing documents: {str(e)}")

    # Check vector store status
    if os.path.exists("chroma_index"):
        st.success("✅ Vector store exists")
    else:
        st.warning("⚠️ No vector store found - Please index your documents")

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    # Search interface
    st.markdown("### 🔍 Search Documents")
    query = st.text_input("Enter your search query", placeholder="What would you like to know?")
    search_button = st.button("Search")

    if query and search_button:
        with st.spinner("🔎 Searching through documents..."):
            try:
                # Process query using pipeline
                result = st.session_state.pipeline.process_query(query, k=3)
                response = result["response"]
                
                # Display text response
                st.markdown("### 📑 Search Results")
                st.write(response["text_response"])
                
                # Display visualization if available
                if response.get("visualization"):
                    vis_data = response["visualization"]
                    chart_type = vis_data["type"].lower()
                    
                    try:
                        if chart_type == "bar":
                            fig = go.Figure(data=[
                                go.Bar(x=vis_data["labels"], y=vis_data["data"])
                            ])
                        elif chart_type == "line":
                            fig = go.Figure(data=[
                                go.Scatter(x=vis_data["labels"], y=vis_data["data"], mode='lines+markers')
                            ])
                        elif chart_type == "pie":
                            fig = go.Figure(data=[
                                go.Pie(labels=vis_data["labels"], values=vis_data["data"])
                            ])
                        
                        fig.update_layout(
                            title=vis_data["title"],
                            template="plotly_white",
                            margin=dict(t=50, l=50, r=50, b=50)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                
                # Display confidence score
                if "confidence_score" in response:
                    confidence = response["confidence_score"]
                    st.progress(confidence, text=f"Confidence: {confidence:.0%}")
                
                # Display relevant documents
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(result["relevant_documents"], 1):
                        st.markdown(f"""
                        <div class="search-result">
                            <h4>Document {i}</h4>
                            <p>{doc.page_content[:500]}...</p>
                            <small>Source: {doc.metadata.get('source', 'Unknown')}</small>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error searching documents: {str(e)}")
                if "No vector store found" in str(e):
                    st.info("Please index your documents first using the button in the sidebar.")

with col2:
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    user_question = st.text_input("Ask a question about the documents", placeholder="Ask me anything...")
    ask_button = st.button("Ask")

    if user_question and ask_button:
        with st.spinner("🤔 Thinking..."):
            try:
                # Process query using pipeline
                result = st.session_state.pipeline.process_query(user_question, k=2)
                response = result["response"]
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("assistant", response["text_response"]))
                
                # Display visualization if available
                if response.get("visualization"):
                    vis_data = response["visualization"]
                    chart_type = vis_data["type"].lower()
                    
                    try:
                        if chart_type == "bar":
                            fig = go.Figure(data=[
                                go.Bar(x=vis_data["labels"], y=vis_data["data"])
                            ])
                        elif chart_type == "line":
                            fig = go.Figure(data=[
                                go.Scatter(x=vis_data["labels"], y=vis_data["data"], mode='lines+markers')
                            ])
                        elif chart_type == "pie":
                            fig = go.Figure(data=[
                                go.Pie(labels=vis_data["labels"], values=vis_data["data"])
                            ])
                        
                        fig.update_layout(
                            title=vis_data["title"],
                            template="plotly_white",
                            margin=dict(t=50, l=50, r=50, b=50)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                
                # Display confidence score
                if "confidence_score" in response:
                    confidence = response["confidence_score"]
                    st.progress(confidence, text=f"Confidence: {confidence:.0%}")
                
                # Display sources
                if "sources" in response:
                    with st.expander("Sources"):
                        for source in response["sources"]:
                            st.text(f"• {source}")
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                if "No vector store found" in str(e):
                    st.info("Please index your documents first using the button in the sidebar.")

    # Display chat history
    st.markdown("### Chat History")
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")

# Footer
st.markdown("---")
st.markdown("*Powered by INGRES-AI*")