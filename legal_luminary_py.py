import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from sentence_transformers import SentenceTransformer
from langchain.chains import load_qa_chain
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

# Accessing the API keys from Streamlit secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_api_env = st.secrets["PINECONE_API_ENV"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Ensure all keys are retrieved successfully
if not all([pinecone_api_key, pinecone_api_env, openai_api_key]):
    st.error("API keys are missing. Please check the environment variables.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Create or access the index
index_name = 'legal'
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_api_env
        )
    )

index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text_field")

# Initialize models and chain only once
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
llm = OpenAI(api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

# Streamlit interface layout
st.set_page_config(page_title="Singaram's Legal Advisor", layout="wide")

# Hide the footer and menu
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title {
            display: flex;
            align-items: center;
        }
        .sidebar-title img {
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="sidebar-title">
            <img src="https://raw.githubusercontent.com/Vikas-Singaram/ShockerBot/3a323888507453b5639e7168f79c16cf9ed2ba39/wsu_logo.png" width="30" />
            <h1>ShockerBot</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("Navigation")
    nav_options = ["Home", "About", "Settings"]
    nav_choice = st.radio("Go to", nav_options)
    
    if st.button("Clear History"):
        if "history" in st.session_state:
            st.session_state.history = []
        st.experimental_rerun()

# Custom CSS for fixed input bar and smaller button
st.markdown(
    """
    <style>
    .input-bar {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: white;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        display: flex;
        z-index: 9999;
    }
    .input-bar input[type='text'] {
        flex: 1;
        margin-right: 10px;
        padding: 5px;
        font-size: 16px;
    }
    .input-bar button {
        padding: 5px 10px;
        font-size: 16px;
    }
    .title-bar {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .title-bar img {
        margin-right: 10px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

if nav_choice == "Home":
    st.markdown(
        """
        <div class="title-bar">
            <img src="https://raw.githubusercontent.com/Vikas-Singaram/ShockerBot/3a323888507453b5639e7168f79c16cf9ed2ba39/wsu_logo.png" width="70" />
            <h1>ShockerBot</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Question bar at the top of the screen
    st.write('<div class="input-bar">', unsafe_allow_html=True)
    
    with st.form(key='question_form', clear_on_submit=True):
        user_query = st.text_input("Enter your question:", key="input_box")
        submit_button = st.form_submit_button(label='Send')
        
        if submit_button and user_query:
            with st.spinner('Processing...'):
                # Save user input to history
                st.session_state.history.insert(0, {"role": "user", "content": user_query})

                # Optimize query retrieval
                query_vector = model.encode(user_query).tolist()
                docs = vectorstore.query(vector=query_vector, top_k=5, include_metadata=True)

                # Convert Pinecone results to Document objects
                docs_list = []
                for doc in docs['matches']:
                    if 'metadata' in doc and 'text' in doc['metadata']:
                        docs_list.append(Document(page_content=doc['metadata']['text'], metadata=doc['metadata']))
                    else:
                        st.write(f"Skipping a doc due to missing metadata or text: {doc}")

                # Prepare input data
                input_data = {
                    'input_documents': docs_list,
                    'question': user_query,
                }

                # Invoke the chain and handle the response
                try:
                    response = chain.run(input_data)
                    st.session_state.history.insert(1, {"role": "bot", "content": response})
                except Exception as e:
                    st.session_state.history.insert(1, {"role": "bot", "content": f"Encountered an error: {e}"})
    
    st.write('</div>', unsafe_allow_html=True)

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.history:
            if message["role"] == "user":
                st.write(f"<div style='text-align: right; margin: 10px 0;'><img src='https://img.icons8.com/ios-glyphs/30/000000/user.png' style='vertical-align:middle; margin-right: 5px;'></div><div style='text-align: right; margin: 10px 0;'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.write(f"<div style='text-align: left; margin: 10px 0;'><img src='https://raw.githubusercontent.com/Vikas-Singaram/ShockerBot/26dc1c1a93e4a5dc9c5fbfa81603da64cfa23731/shocker_bot_logo.jpg' style='vertical-align:middle; margin-right: 5px;' width='30'></div><div style='text-align: left; margin: 10px 0;'>{message['content']}</div>", unsafe_allow_html=True)

elif nav_choice == "About":
    st.title("About")
    st.write("This is a ChatGPT-like interface built with Streamlit.")
    st.write("You can ask questions and get responses from a language model.")

elif nav_choice == "Settings":
    st.title("Settings")
    st.write("Settings page for configuring the app.")
