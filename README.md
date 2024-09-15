
# Basic-RAG-using-Langchain

This project implements a basic Retrieval-Augmented Generation (RAG) system using Langchain, a framework for building applications that integrate language models with knowledge bases and other data sources.

## Features
- Utilizes the Langchain framework to build a RAG system.
- Integrates with OpenAI's API for language model interactions.
- Includes document retrieval using FAISS for efficient search.
- Supports PDF document parsing with PyPDF.
- Built with Streamlit for easy deployment of web applications.

## Requirements
Before running the project, ensure you have the following dependencies installed:

```bash
langchain
streamlit
pypdf
langchain-community
langchain-core
faiss-cpu
tiktoken
openai
```

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
    ```bash
    cd Basic-RAG-using-Langchain
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

4. The app will be available on your local server. You can interact with the system to input questions and receive responses from the RAG model.

## File Structure
- **main.py**: The main application script that integrates Langchain and OpenAI APIs to build the RAG system.
- **Dockerfile**: Contains instructions for containerizing the project for deployment.
- **requirements.txt**: Lists all the required packages to run the application.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Langchain](https://github.com/hwchase17/langchain) for providing the framework for building RAG systems.
- [OpenAI](https://openai.com/) for the language model.
