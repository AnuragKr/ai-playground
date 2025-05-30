from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import os

# Create an instance of FastAPI to serve as the main application.
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchain's Runnable interfaces",
)

# Configure CORS middleware to allow all origins, enabling cross-origin requests.
# details: https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/liveness")
def liveness():
    """
    Define a liveness check endpoint.

    This route is used to verify that the API is operational and responding to requests.

    Returns:
        A simple string message indicating the API is working.
    """
    return 'API Works!'


# create input prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Act as a helpful AI assistant, answer questions in detail with examples as necessary"),
        ("human", "{input}"),
    ]
)

# Initialize the OpenAI Chat instance with specific model parameters.
chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# create simple llm chain
llm_chain = (prompt
                |
             chatgpt
                |
             StrOutputParser()
)


# Register routes using LangChain's utility function which integrates the chat model into the API.
add_routes(
    app,
    llm_chain,
    path="/llm_chain",
)

if __name__ == "__main__":
    import uvicorn
    # Start the server on localhost at port 8989.
    uvicorn.run(app, host="127.0.0.1", port=8989)
