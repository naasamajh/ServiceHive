# 🚀 AutoStream Social-to-Lead Agentic Workflow

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![LangGraph](https://img.shields.io/badge/Architecture-LangGraph-orange.svg)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3-black.svg)
![Rich Interface](https://img.shields.io/badge/UI-Python_Rich-blueviolet.svg)

An enterprise-grade, highly advanced AI conversational agent built for **AutoStream** (a fictional SaaS video editing platform). This application flawlessly bridges the gap between casual customer support and high-intent lead generation, using state-of-the-art agentic workflows.

---

## ✨ Advanced Features

1. **Native Token Streaming**: The CLI utilizes asynchronous chunk broadcasting to visually type out the LLM's thought process in real-time, completely eliminating loading silences.
2. **Pydantic Structured Output Router**: Eliminates routing anomalies by strictly forcing Llama 3 to output categorized JSONs via `.with_structured_output`, enabling mathematically perfect intent determination.
3. **In-Memory RAG VectorStore**: Uses `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) populated with rich, comprehensive Markdown knowledge to eliminate LLM pricing/feature hallucinations.
4. **Resilient Graph Memory**: Anchored with LangGraph's `MemorySaver` checkpointer, history is flawlessly preserved allowing the user to contextually bounce between RAG inquiries and lead captures seamlessly.
5. **Gorgeous CLI Interface**: Styled intricately utilizing the `Rich` framework, creating formatted markdown panels, colors, and contextual developer debugging logs right in the console.

---

## ⚙️ Technical Stack Blueprint

- **Orchestration**: LangGraph (for explicit state machines & memory)
- **Intelligence**: Groq (`llama-3.3-70b-versatile`)
- **Embeddings**: Local HuggingFace Transformers (SentenceTransformers)
- **Database**: In-Memory Facebook AI Similarity Search (FAISS) 
- **Presentation Layer**: Python `rich` Library

---

## 🛠️ How to Launch Locally

### 1. Prerequisites
- Python 3.9+
- A completely free [Groq API Key](https://console.groq.com/)

### 2. Environment Setup
```bash
# Clone and enter the directory
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate

# Install the necessary dependencies
pip install -r requirements.txt
```

### 3. API Key Binding
Create a `.env` file in the base folder with:
```env
GROQ_API_KEY="gsk_your_api_key_here..."
```

### 4. Execute the App
```bash
python main.py
```

---

## 🏛️ Architecture & State Explanation

**Why skip basic LangChain tools for LangGraph?**
Basic prompt tool chains fail at maintaining rigid boundaries when conversational dynamics diverge radically. By structuring this application as a **Directed Graph**, we establish explicit node-based "Modes":

1. **The Intention Gatekeeper (`intent_detector`)**: Every single message natively hits this node first. Because we pass a strict Pydantic `BaseModel` schema to the LLM here, it mechanically restricts generated content to yielding *only* permitted intents—serving as an unbreakable structural guardrail.
2. **The Reactive Core (`agent`)**: Depending on the Router's classification, this central node shifts its operational persona—either executing FAISS RAG bindings rapidly or locking down conversational flows until missing constraints (Lead Details) are aggressively sourced.
3. **Persisted State (`MemorySaver`)**: Conversation tracking isn't done via manually appending strings to arrays. The thread state is mapped globally by LangGraph checkpointers indexed strictly via a `thread_id`. 

---

## 📱 WhatsApp Webhook Deployment Strategy

To deploy this intelligent Agent to seamlessly resolve queries on the **Meta WhatsApp Cloud API**:
1. **API Ingress Setup**: Launch a FastAPI/Flask microservice exposing an HTTP `POST` route mapped directly to your Meta Developer App webhook configuration.
2. **Platform Authentication**: Rig the endpoint to resolve Meta's `hub.verify_token` automated `GET` handshake.
3. **Payload Destructuring**: Extract `entry[0].changes[0].value.messages[0].text.body` and the user's `wa_id` (phone number identifier) from incoming POST JSON payloads.
4. **State Restoration**: Pass the user's phone number straight into the LangGraph executor via the `config` block (`config={"configurable": {"thread_id": phone_number}}`). LangGraph will inherently resume that specific customer's timeline instantly!
5. **Egress Pipeline**: As the LLM completes the generated tokens, package the text string directly back into a routine HTTP `POST` push to the Graph API endpoint (`https://graph.facebook.com/v21.0/{phone_number_id}/messages`), arriving on their phone!

---

## 🎥 Recording your Demo Video

The system explicitly prints metadata debugging visualizations to flawlessly broadcast execution rules across 2-3 minutes:

1. Provide a standard simple greeting. *(Console Output detects: `Casual greeting`)*. 
2. Aggressively test the RAG by querying specifications like '4K resolution limits and audio tools'. *(Console detects: `Product Inquiry`, triggering the `⚙️ System: Searching RAG` logic visually).*
3. Pivot cleanly! State explicitly you want to subscribe to the Pro plan for your YouTube. *(Console triggers: `High-intent lead (ready to sign up)`, causing the workflow to lock out generic answers and solicit your Name and Email).*
4. Answer the prompt. *(The Graph completes the schema trigger, resolving the constraint and finally printing a spectacular Green CRM Success Panel!)*
