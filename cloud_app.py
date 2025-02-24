# --- Imports ---
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import streamlit as st
import re
import os


##### BOT SETUP #####

# Get API token from streamlit secrets and set it globally
os.environ["HUGGINGFACEHUB_API_TOKEN"]=st.secrets['HUGGINGFACEHUB_API_TOKEN'] 

# Set up LLM
llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    task='text-generation',
    max_new_tokens=512,
    temperature=0.01,
    top_p=0.95,
    repetition_penalty=1.03
)
# Set up embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    cache_folder="embeddings_cache/"
)
# Load Vector Database
vector_db = FAISS.load_local(
    "data/faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)
# Set up RAG retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# Setting Up Customizable System Prompt:

# Mood and factuality settings (how to alter the prompt based on the user's inputs)
moods_dict = {
    'Expert': { 
        #there are two parts of the prompt I want to alter in response to one input
        'tone': 'nice',
        'response': 'concise and factual'},
    'Rude': {
        'tone': 'rude',
        'response': 'sassy and confrontational'}
}
factuality_dict = {
    'Context only': "based only on the following context and previous conversation. ",
    'Improvise': "primarily based on the following context and previous conversation but use your judgement if the context doesn't seem to have the answer."
}
# Initialize mood and factuality in session state
if "mood" not in st.session_state:
    st.session_state.mood = 'Expert'
if "factuality" not in st.session_state:
    st.session_state.factuality = 'Context only'

# Prompt concatonating in customizable aspects
template = "You are a " + moods_dict[st.session_state.mood]['tone'] + "chatbot having a conversation with a human. Answer the question " + factuality_dict[st.session_state.factuality] + """
Keep your response """ + moods_dict[st.session_state.mood]['response'] + """. End your response without predicting additional human questions. 

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {input}
Chatbot's Response:"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


# Setting up cache for bot
@st.cache_resource
def init_bot():
    doc_retriever = create_history_aware_retriever(llm, retriever, prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(doc_retriever, doc_chain)
# Getting bot from cache
rag_bot = init_bot()



##### RESPONSE IMPROVEMENT FUNCTIONS #####

# Removing Unwanted Citations:

# Defining a function for filtering sources 
def citations_filters(doc):
    text = doc.page_content.upper()
    # Don't cite the Gutenberg header
    gut_header = 'GUTENBERG' in text
    # Don't quote the table of contents
    table_of_contents = " . . . . " in text
    # Check filters
    return not (gut_header or table_of_contents)
def remove_unwanted_citations(sources):
    # Excludes "sources" that fail the filter check
    return [doc for doc in sources if citations_filters(doc)]


# Hallucination Prevention Measures:

# Mitigate question continuation
def ensure_punctuation(user_message):
    # The bot was adding more to the user's question when they didn't use punctuation.
    # By ensuring the bot sees a completed sentence we can mitigate this problem
    if user_message[-1] not in ['.', '!', '?']:
        return user_message + '.'
    else: return user_message
# Remove hallucinated questions before displaying or adding them to memory
def remove_hallucinations(response):
    return response.split(' Human:')[0]
# Remove "Assistant: " at the start as well as any strange leading spaces and punctuation (problems I was seeing)
def clean_up_response(response):
    match = re.search(r'^(?:\s*(?:Bot: |AI: |Assistant: |Chatbot: )?\s*[\.,!?-]*\s*)?(.*?)?$', response)
    return match.group(1) if match else response



##### STREAMLIT #####

st.title("Statistical Learning in Wonderland")


# --- System Prompt Customization ----

# Select Mood
st.session_state.mood = st.sidebar.selectbox(
    label = 'Chatbot Tone',
    options = ['Expert', 'Rude'],
    index = 0,
    help = "Can adjust the tone of the chatbot's responses")
# Select Factuality
st.session_state.factuality = st.sidebar.selectbox(
    label = 'Chatbot Factuality',
    options = ['Context only', 'Improvise'],
    index = 0,
    help = "Can adjust whether the bot relies solely on the citable docs")


# --- Chat ----

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.sources = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user message
if user_message := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(user_message)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Alice is performing a similarity search through the vector database..."):

        # Send question to bot to get answer
        answer = rag_bot.invoke({"input": ensure_punctuation(user_message), 
                                 "chat_history": st.session_state.messages, 
                                 "context": retriever})

        # Clean up response format and remove hallucinations
        response = clean_up_response(remove_hallucinations(answer['answer']))
        print(answer['answer'])

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history (make sure this is the one with the hallucinations removed)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.sources.append(remove_unwanted_citations(answer['context']))


# --- Source Citation ----

# Doc names for displaying sources
doc_names = {
    'docs/alice_in_wonderland.txt':'Alice in Wonderland by Lewis Caroll',
    'docs/ISLP_website.pdf':'An Introduction to Statistical Learning'
}
# If the bot has sent a message, you can ask it to cite its sources
if st.session_state.messages:
    if st.button("Cite source"):
        st.sidebar.markdown('# Citations:')

        recent_sources = st.session_state.sources[-1]
        # Say if there are no sources (removing unwanted citations could cause this)
        if len(recent_sources) == 0:
            st.sidebar.write('No relevant sources found')
        else:
            # Loop through most recent sources to reformat and display them
            for doc in recent_sources:
                source = doc.metadata['source']
                source_name = doc_names[source] #reformat name using dict above
                quote = doc.page_content
                # Display formated with markdown (including putting quote in italics)
                st.sidebar.markdown(f'## Source: \n{source_name} \n## Quote: \n"*{quote}*"')