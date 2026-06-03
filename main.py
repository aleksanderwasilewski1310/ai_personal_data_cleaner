import os
import uuid
from dotenv import load_dotenv
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# LangChain and LangGraph ecosystem imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

# Set Model
LLM = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

app = FastAPI(title="LangGraph Personal Data Parser API")

# --- 1. Pydantic Schemas for Data Validation and API Contracts ---


class PersonalData(BaseModel):
    """Target schema for structured personal data extraction."""

    first_name: Optional[str] = Field(None, description="The individual's given name")
    last_name: Optional[str] = Field(None, description="The individual's family name")
    address: Optional[str] = Field(
        None, description="Street name, house, and apartment number"
    )
    city: Optional[str] = Field(None, description="City or municipality")
    date_of_birth: Optional[str] = Field(
        None, description="Date of birth, preferably in YYYY-MM-DD format"
    )
    country: Optional[str] = Field(None, description="Country of residence")


class GraphState(BaseModel):
    """Represents the shared state of the LangGraph workflow."""

    input_text: str
    parsed_data: Dict[str, Any] = Field(default_factory=dict)
    feedback: Optional[str] = None
    approved: bool = False


# HTTP Request Payloads
class StartProcessRequest(BaseModel):
    text: str


class ReviewRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: Optional[str] = None


# --- 2. Graph Node Definitions ---


def parse_input_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 1: Extracts and maps unstructured raw text into a structured dictionary.
    Incorporates historical state and user feedback if routing back from a rejection.
    """
    structured_llm = LLM.with_structured_output(PersonalData)

    # Constructing a prompt that dynamically handles the initial run vs feedback loops
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an advanced AI data extraction assistant. Your task is to extract "
                    "personal data from raw, unstructured text and map it accurately to the requested schema.\n"
                    "If 'Previous Attempt' and 'User Feedback' are provided, refine the extraction "
                    "by addressing the issues highlighted in the feedback."
                ),
            ),
            (
                "human",
                (
                    "Raw input text: {input_text}\n\n"
                    "Previous attempt (if any): {parsed_data}\n"
                    "User feedback (if any): {feedback}"
                ),
            ),
        ]
    )

    chain = prompt | structured_llm
    response = chain.invoke(
        {
            "input_text": state.input_text,
            "parsed_data": state.parsed_data,
            "feedback": state.feedback,
        }
    )

    # Update the graph state with the newly extracted dictionary
    return {"parsed_data": response.model_dump()}


def human_review_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 2: Implements Human-in-the-Loop (HITL) by raising an execution interrupt.
    The graph state is persisted and frozen until resumed via an external API call.
    """
    # The interrupt() function halts execution and exposes the state to the client.
    # When resumed, the payload passed into Command(resume=...) becomes the return value.
    user_response = interrupt(
        {
            "message": "Human verification required for the extracted data.",
            "current_data": state.parsed_data,
        }
    )

    # Map the external payload back into the graph state
    return {
        "approved": user_response["approved"],
        "feedback": user_response.get("feedback"),
    }


# --- 3. Conditional Routing Logic (Edges) ---


def route_after_review(state: GraphState):
    """
    Evaluates the state after human intervention to determine the next execution path.
    """
    if state.approved:
        return END
    return "parse_input_node"


# --- 4. Graph Construction and Compilation ---

workflow = StateGraph(GraphState)

# Registering nodes to the graph blueprint
workflow.add_node("parse_input_node", parse_input_node)
workflow.add_node("human_review_node", human_review_node)

# Defining execution flow edges
workflow.add_edge(START, "parse_input_node")
workflow.add_edge("parse_input_node", "human_review_node")

# Conditional routing out of the Human Review node based on user approval
workflow.add_conditional_edges(
    "human_review_node",
    route_after_review,
    {END: END, "parse_input_node": "parse_input_node"},
)

# MemorySaver checkpointer acts as the state store across async stateless HTTP requests
memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)


# --- 5. FastAPI Endpoint Handlers ---


@app.post("/process/start")
async def start_processing(payload: StartProcessRequest):
    """
    Initializes the stateful graph execution thread.
    Runs through Node 1 and halts inside Node 2 due to the active interrupt.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"input_text": payload.text}

    # Execute the graph. It will pause automatically at the human_review_node interrupt.
    current_state = graph_app.invoke(initial_state, config)

    # Fetch current checkpoint metadata to confirm the interrupt status
    state_info = graph_app.get_state(config)

    return {
        "thread_id": thread_id,
        "status": "Awaiting Review",
        "parsed_data": current_state.get("parsed_data"),
        "interrupt_details": state_info.next,  # Target node waiting for execution
    }


@app.post("/process/review")
async def review_processing(payload: ReviewRequest):
    """
    Resumes a frozen graph thread with human input (approval or correction feedback).
    """
    config = {"configurable": {"thread_id": payload.thread_id}}

    # Verify if the thread exists and is currently waiting on an interrupt
    state_info = graph_app.get_state(config)
    if not state_info.next:
        raise HTTPException(
            status_code=400,
            detail="This thread has already finished or does not exist.",
        )

    # Package the human input into a Command object to resume the graph execution
    resume_action = Command(
        resume={"approved": payload.approved, "feedback": payload.feedback}
    )

    # Resume the workflow precisely where it was suspended
    final_state = graph_app.invoke(resume_action, config)

    # Re-evaluate the new checkpoint state post-execution
    new_state_info = graph_app.get_state(config)

    if not new_state_info.next:
        # The graph reached the END state successfully
        return {
            "status": "Completed",
            "message": "Data successfully approved and verified.",
            "final_data": final_state.get("parsed_data"),
        }
    else:
        # The graph routed back to parse_input_node and is halted again with new data
        return {
            "status": "Returned for correction",
            "message": "Data rejected by user. LLM is re-processing based on provided feedback.",
            "current_data": final_state.get("parsed_data"),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
