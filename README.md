
This project implements a **Retrieval-Augmented Generation (RAG)** model that combines **document retrieval** with **text generation** for answering user queries. The system uses **LangChain** and **Chainlit** to provide an interactive chatbot interface that retrieves relevant documents from a set of text files and generates context-aware responses. The core functionality is based on the **FLAN-T5 model** for text generation and **sentence-transformers** for document retrieval.

### Key Features
1. **Document Retrieval and Answer Generation**: 
   - The system retrieves relevant documents from a specified directory using **sentence-transformers** and generates answers with the **FLAN-T5** model.
   - The retrieval process ensures that the answers are based on the relevant documents available.

2. **Interactive Chatbot**: 
   - The **Chainlit** interface allows users to interact with the system. The chatbot receives questions, retrieves relevant context from the documents, and generates responses.

3. **Dynamic History Management**:
   - **ConversationBufferMemory** stores the user's conversation history, which is used by the model to improve response relevance over multiple interactions.
  
4. **Prompt Engineering**:
   - The model is prompted with a structured template that includes both the history of the conversation and the retrieved context, ensuring more accurate and relevant answers.

5. **Error Handling**:
   - The system includes robust error handling to manage cases where no relevant documents are found or other issues arise during processing.

6. **Text Splitting and Embeddings**:
   - Documents are split into smaller chunks for better processing, and embeddings are generated using **HuggingFaceEmbeddings**, which are stored in a **FAISS** index for efficient retrieval.

### How It Works
1. **Text Document Loading**: 
   - Text files from a specified directory are loaded using **TextLoader** and split into smaller chunks using **CharacterTextSplitter**.
  
2. **Document Embeddings**:
   - The chunks are embedded using **HuggingFaceEmbeddings** and stored in **FAISS**, enabling fast retrieval of relevant documents based on a user query.

3. **Answer Generation**:
   - Once relevant documents are retrieved, the **FLAN-T5** model is used to generate answers based on both the user's question and the context.

4. **Interactive User Experience**:
   - Through **Chainlit**, users send queries, which are processed and answered dynamically by the model. The conversation history is maintained to ensure context-aware responses.

### Flow
- The system is initialized with a **LLMRAGModel**, which handles document loading, text splitting, embedding creation, and the retrieval process.
- When the user interacts with the chatbot, the system:
  1. Retrieves relevant documents based on the query.
  2. Uses the **LLM chain** to generate a response by considering both the query and the retrieved context.
  3. Returns the response to the user.

### Setup & Usage
1. Clone the repository and install the necessary dependencies.
2. Set up the document directory where your text files are stored.
3. Start the chatbot through **Chainlit**, interact with it, and ask questions based on the content of the documents.

