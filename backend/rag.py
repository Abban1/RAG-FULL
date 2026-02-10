import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from groq import Groq

from pdf_loader import load_and_split_pdf

load_dotenv()

PDF_PATH = "data/document.pdf"
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "ielts"
pages_split = load_and_split_pdf(PDF_PATH)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

vectorstore = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


@tool
def retriever_tool(query: str) -> str:
    """
    Search the uploaded PDF documents for relevant information based on the user's query.
    Use this tool whenever you need specific context from the provided files.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in the PDF."

    return "\n\n".join(doc.page_content for doc in docs)


class GroqLLM:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY missing.")
        self.client = Groq(api_key=api_key)

    def invoke(self, messages: list):
        groq_messages = []
        for m in messages:
            role = "system" if isinstance(m, SystemMessage) else "user"
            groq_messages.append({"role": role, "content": m.content})

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant", temperature=0.7, messages=groq_messages
        )
        return HumanMessage(content=response.choices[0].message.content)


llm = GroqLLM(api_key=os.environ.get("GROQ_API_KEY"))


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: AgentState) -> dict:
    messages = list(state["messages"])
    user_query = messages[-1].content

    context = retriever_tool.invoke(user_query)

    system_prompt = (
        "You are an expert answer giver who reads PDF files. "
        "Use ONLY the provided context to answer.\n\n"
        f"CONTEXT:\n{context}"
    )

    prompt = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(prompt)

    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

rag_agent = workflow.compile()


def ask_rag(question: str) -> str:
    result = rag_agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
