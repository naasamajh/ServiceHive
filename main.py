import os
import sys
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# Rich framework for high-quality terminal visuals
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()

# Load environment variables
load_dotenv()

# --- 1. RAG Setup ---
def setup_rag():
    kb_path = "knowledge_base.md"
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"{kb_path} not found!")
    
    loader = TextLoader(kb_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Using local HuggingFace embeddings
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return vectorstore.as_retriever()

retriever = None
try:
    retriever = setup_rag()
except Exception as e:
    console.print(f"[bold red]Warning: RAG setup failed. Details: {e}[/bold red]")

# --- 2. Tools ---
@tool
def search_knowledge_base(query: str) -> str:
    """
    RAG-Powered Knowledge Retrieval.
    Use this RAG tool to fetch information about AutoStream Pricing, Features, and Company Policies.
    """
    console.print(f"\n[cyan]⚙️  System:[/cyan] [dim]Searching RAG Knowledge Base for: '{query}'...[/dim]")
    if retriever:
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs])
    else:
        try:
            with open("knowledge_base.md", "r") as f:
                return f.read()
        except:
            return "Knowledge base unavailable."

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Execute Lead Capture. Call this tool ONLY after collecting exactly these three values from an interested user:
    1. Name
    2. Email
    3. Creator Platform (YouTube, Instagram, etc.)
    Do NOT call this prematurely!
    """
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]*** LEAD CAPTURED SUCCESSFULLY ***[/bold green]\n\n"
        f"[bold]Name:[/bold]     {name}\n"
        f"[bold]Email:[/bold]    {email}\n"
        f"[bold]Platform:[/bold] {platform}",
        title="CRM Action", border_style="green"
    ))
    return "Lead capture successful. Next step: Enthusiastically confirm with the user that their details were saved."

tools = [search_knowledge_base, mock_lead_capture]

# --- 3. State & Graph Setup ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str

class IntentClassification(BaseModel):
    intent: Literal["Casual greeting", "Product or pricing inquiry", "High-intent lead (ready to sign up)", "Unknown"] = Field(
        description="The primary explicitly detected intent of the user's latest conversational message."
    )

def intent_detector(state: AgentState):
    """Detects intent of the latest user message utilizing Strict Structured JSON Output."""
    messages = state["messages"]
    
    user_inputs = [m.content for m in messages if isinstance(m, HumanMessage)]
    if not user_inputs:
        return {"intent": "Unknown"}
        
    latest_input = user_inputs[-1]
    
    # Utilizing structured output strictly enforces mathematical precision vs string parsing.
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    try:
        response = structured_llm.invoke(f"Classify this message functionally: '{latest_input}'")
        intent = response.intent
    except Exception:
        intent = "Product or pricing inquiry" # Fallback heuristic
        
    # Exposing the intent thought process dynamically in the CLI
    console.print(f"[magenta]🧠 Graph Intent Router Detected:[/magenta] [bold italic]{intent}[/bold italic]")
    
    return {"intent": intent}
    
def agent_node(state: AgentState):
    """Main intelligent agent generating token responses natively."""
    messages = state["messages"]
    intent = state.get("intent", "Unknown")
    
    # We execute generation using Llama with streaming parameters enabled!
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = f"""
    You are the primary sales and support AI for AutoStream, a premium SaaS video editing suite.
    
    The user's current conversational intent is: "{intent}"
    
    Instructions:
    1. For greetings: Be warm, friendly, and briefly introduce AutoStream.
    2. For product or pricing inquiries: Use the search_knowledge_base tool to securely retrieve the exact specifications. Do not guess prices.
    3. For high-intent signups: You must collect exactly three details natively in conversation: Name, Email, and Creator Platform. Once all three are provided by the user, execute the mock_lead_capture tool.
      
    Rule: Never mention your tool names to the user. Always use clean markdown.
    """
    
    sys_msg = SystemMessage(content=system_prompt)
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": [response]}
    
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return END

# Build the Execution Graph
builder = StateGraph(AgentState)

builder.add_node("intent_detector", intent_detector)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "intent_detector")
builder.add_edge("intent_detector", "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

# Enable memory mapping configuration
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- 4. Interactive Advanced CLI Interface ---
def chat():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]AutoStream Conversational AI[/bold cyan]\n"
        "[dim]Powered by LangGraph, HuggingFace RAG, & Groq Llama 3[/dim]\n\n"
        "To terminate the session, type [bold red]'quit'[/bold red].", 
        title="Agent Workspace", border_style="cyan"
    ))
    
    if not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]CRITICAL EXCEPTION:[/bold red] GROQ_API_KEY environment variable is currently absent.")
        sys.exit(1)
        
    # Maintains coherent conversational continuity
    config = {"configurable": {"thread_id": "demo_session"}}
        
    while True:
        try:
            # Visually formatted prompt
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\n[dim]Session gracefully terminated.[/dim]")
            break
            
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values"  # Reverting to values mode to bypass Groq Llama 3 streaming JSON bug
        )
        
        for event in events:
            messages = event.get("messages", [])
            if messages:
                latest_msg = messages[-1]
                if isinstance(latest_msg, AIMessage) and latest_msg.content:
                    console.print(f"\n[bold purple]Agent Feedback:[/bold purple]\n{latest_msg.content}")

if __name__ == "__main__":
    chat()
