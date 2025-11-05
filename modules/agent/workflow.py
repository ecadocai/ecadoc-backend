"""
Agent workflow and memory management for the Floor Plan Agent API
"""
import uuid
import random
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from modules.config.settings import settings
from modules.agent.tools import (
    load_pdf_for_floorplan,
    convert_pdf_page_to_image,
    detect_floor_plan_objects,
    verify_detections,
    internet_search,
    generate_frontend_annotations,
    answer_question_using_rag,
    answer_question_with_suggestions,
    analyze_pdf_page_multimodal,
    measure_objects,
    calibrate_scale,
    analyze_object_proportions,
    clean_temp_image
)

# List of all available tools
ALL_TOOLS = [
    load_pdf_for_floorplan,
    convert_pdf_page_to_image,
    detect_floor_plan_objects,
    verify_detections,
    internet_search,
    generate_frontend_annotations,
    answer_question_using_rag,
    answer_question_with_suggestions,
    analyze_pdf_page_multimodal,
    measure_objects,
    calibrate_scale,
    analyze_object_proportions,
    clean_temp_image
]
from modules.session import session_manager, context_resolver
from modules.database.models import db_manager

# ==============================
# Memory Management
# ==============================

class SimpleChatMessageHistory:
    """Simple chat message history implementation"""
    def __init__(self):
        self.messages = []
    
    def add_message(self, message):
        self.messages.append(message)
    
    def clear(self):
        self.messages = []

class SimpleMemory:
    """Simple memory implementation to store conversation history"""
    def __init__(self):
        self.chat_memory = SimpleChatMessageHistory()
        self.memory_key = "history"
    
    def load_memory_variables(self, inputs):
        return {self.memory_key: "\n".join([str(m) for m in self.chat_memory.messages])}
    
    def save_context(self, inputs, outputs):
        # Extract human input
        human_input = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
        # Extract AI output
        ai_output = outputs.get("output", "") if isinstance(outputs, dict) else str(outputs)
        
        if human_input:
            self.chat_memory.add_message(HumanMessage(content=human_input))
        if ai_output:
            self.chat_memory.add_message(AIMessage(content=ai_output))
    
    def clear(self):
        self.chat_memory.clear()

# ==============================
# LangChain Memory Store (per-session)
# ==============================

# This dictionary acts as an in-memory store for chat histories,
# keyed by a session_id. It enables RunnableWithMessageHistory to
# maintain per-session context across invocations.
_memory_store: Dict[str, ChatMessageHistory] = {}

def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ChatMessageHistory()
    return _memory_store[session_id]

# ==============================
# Enhanced Session Management
# ==============================

@dataclass
class ChatMessage:
    """Chat message data structure (kept for backward compatibility)"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

# ==============================
# Agent State and Workflow
# ==============================

class FloorPlanState(MessagesState):
    """Represents the state of our floor plan annotation workflow"""
    pdf_path: str
    output_path: str
    annotation_type: str = ""
    temp_image_path: str = ""
    detected_objects: List[Dict] = field(default_factory=list)
    page_number: int = 1

class AgentWorkflow:
    """Agent workflow using LangGraph with proper tool calling"""
    
    def __init__(self):
        self.memory = SimpleMemory()
        self.session_manager = session_manager
        self.context_resolver = context_resolver

        # Use model name from settings so it can be configured via environment variable OPENAI_MODEL
        self.llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.0, api_key=settings.OPENAI_API_KEY)
        self.agent = create_tool_calling_agent(self.llm, ALL_TOOLS, self._create_prompt())
        self.agent_executor = AgentExecutor(agent=self.agent, tools=ALL_TOOLS, verbose=True)

        # Wrap with per-session message history
        self.agent_executor_w_memory = RunnableWithMessageHistory(
            self.agent_executor,
            _get_session_history,
            input_messages_key="messages",
            history_messages_key="history",
        )

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        self.compiled_graph = self.workflow.compile()
    
    def _create_prompt(self):
        """Create the agent prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are Ecadoc AI, an intelligent blueprint assistant for construction documents and floor plans. Understand the user's goal, choose the best tool chain, and return the output in the correct format.

IDENTITY AND GREETINGS
- If the user greets you or asks who you are, respond briefly: "Hello! I'm Ecadoc AI, an intelligent blueprint assistant. How can I help with your document?"

INTENT AND WORKFLOWS

A) Annotation (visual markup)
- Keywords: highlight, circle, annotate, mark, box, count.
- Two-step workflow: convert_pdf_page_to_image → detect_floor_plan_objects → generate_frontend_annotations (with filter if requested, e.g., 'door').
- Output: if annotations were generated successfully, return ONLY the raw JSON string from generate_frontend_annotations. No extra prose.
- If no matching objects were found: reply conversationally listing available object classes and ask which to annotate. Do NOT return JSON in this case.

B) Measurement (dimensions)
- Keywords: measure, width, height, size, dimensions.
- Workflow: detect_floor_plan_objects → measure_objects.
- Output: conversational with clear numeric values and units.

C) Visual analysis (layout, spatial relationships)
- Keywords: describe layout, what does this page look like, where is X located, spatial arrangement.
- Tool: analyze_pdf_page_multimodal.
- Output: concise, structured natural language.

D) Text Q&A (facts in document text)
- Keywords: specifications, explain notes, legend, tell me about the document.
- Tool hierarchy:
  1) Default: answer_question_with_suggestions (comprehensive answer with related topics and pages).
  2) Simple/direct facts (e.g., project number): answer_question_using_rag (faster, concise).
- Output: natural language is fine. If a tool returns JSON (e.g., from answer_question_with_suggestions), you may return that JSON directly.

E) External or current information
- Keywords: current regulations, latest codes, market prices, recent.
- Tool: internet_search.

RULES
- Never mention internal tool names, file paths, temporary paths, or internal IDs in user-facing text.
- Do not include download URLs or markdown links.
- Use one workflow at a time; do not mix intents in a single answer.
- For annotation: ONLY return raw JSON on success; otherwise use a clear conversational explanation and offer alternatives.
"""),
            MessagesPlaceholder(variable_name="history"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def _create_workflow(self):
        """Create the LangGraph workflow"""
        def should_continue(state: FloorPlanState) -> str:
            return "action" if state["messages"][-1].tool_calls else END

        def call_agent(state: FloorPlanState):
            # Session info for memory scoping
            session_id = state.get("session_id") or "default"
            user_id = state.get("user_id")

            # Get the user's most recent request and build the current-turn message
            original_message = state["messages"][-1].content if state.get("messages") else ""
            current_turn = (
                f"CURRENT TASK:\n"
                f"- PDF Path: {state.get('pdf_path', 'Not set')}\n"
                f"- Page Number: {state.get('page_number', 'Not set')}\n"
                f"- User Request: {original_message}\n\n"
                f"IMPORTANT: If this is an annotation request, follow this workflow:\n"
                f"1) convert_pdf_page_to_image (rasterize the requested page).\n"
                f"2) detect_floor_plan_objects (run on the rasterized image).\n"
                f"3) generate_frontend_annotations (pass detections JSON; include filter if the user asked for a class).\n"
                f"Final output must be ONLY the raw JSON from generate_frontend_annotations when annotations succeed."
            )

            # Prepare input for the agent; history is injected automatically by RunnableWithMessageHistory
            state_with_context = {
                "messages": [HumanMessage(content=current_turn)]
            }

            # Invoke with session-scoped memory
            response = self.agent_executor_w_memory.invoke(
                state_with_context,
                config={"configurable": {"session_id": session_id}},
            )

            return {"messages": [AIMessage(content=response["output"])]}
        workflow = StateGraph(FloorPlanState)
        workflow.add_node("agent", call_agent)
        workflow.add_node("action", ToolNode(ALL_TOOLS))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
        workflow.add_edge("action", "agent")

        return workflow
    
    def process_request(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the agent workflow"""
        try:
            # Enhanced intent detection for generalization
            user_message = initial_state["messages"][-1].content if initial_state.get("messages") else ""
            msg_lower = user_message.lower()
            if any(x in msg_lower for x in ["highlight", "circle", "rectangle", "count", "arrow", "annotate"]):
                intent = "annotation"
            elif any(x in msg_lower for x in ["latest", "current", "recent", "news", "trend", "regulation", "countries", "can i build", "allowed", "permitted", "legal", "law", "code", "standard"]):
                intent = "internet_search"
            elif any(x in msg_lower for x in ["describe", "show", "visual", "layout", "diagram", "where", "located", "appearance", "spatial", "look", "see", "view", "display"]):
                intent = "visual_analysis"
            else:
                intent = "question"

            # Route to correct tool chain
            # All routes use the compiled graph, but intent is passed in context for agent prompt
            final_state = self.compiled_graph.invoke(initial_state, {"recursion_limit": settings.RECURSION_LIMIT, "intent": intent})
            return final_state
        except Exception as e:
            raise Exception(f"Agent workflow error: {str(e)}")
    
    def get_or_create_chat_session(self, session_id: str = None, user_id: int = None, context_type: str = 'GENERAL', context_id: str = None) -> str:
        """Get or create a chat session with context support"""
        # Trigger cleanup occasionally
        if random.random() < settings.SESSION_ACTIVITY_UPDATE_PROBABILITY:
            self.session_manager.cleanup_expired_sessions()
        
        if session_id:
            # Validate existing session
            session = self.session_manager.get_session_by_id(session_id)
            if session and session.is_active:
                # Update activity and return existing session
                self.session_manager.update_session_activity(session_id)
                return session_id
        
        # Create new session if user_id is provided
        if user_id is not None:
            return self.session_manager.get_or_create_session(user_id, context_type, context_id)
        
        # Fallback: create a simple UUID for backward compatibility
        return str(uuid.uuid4())
    
    def get_or_create_context_session(self, user_id: int, context_data: Dict[str, Any]) -> str:
        """Get or create a session based on context data"""
        context_type, context_id = self.context_resolver.resolve_context(context_data)
        return self.session_manager.get_or_create_session(user_id, context_type, context_id)
    
    def add_chat_message(self, session_id: str, role: str, content: str, user_id: int = None):
        """Add a message to chat session with enhanced context support"""
        if user_id is not None:
            # Use the new session manager to add message with context
            success = self.session_manager.add_message_to_session(session_id, user_id, role, content)
            if not success:
                print(f"Warning: Failed to add message to session {session_id}")
        else:
            # Fallback to old memory system for backward compatibility
            if hasattr(self, 'chat_sessions'):
                self.chat_sessions.add_message(session_id, role, content)
    
    def get_chat_history(self, session_id: str, user_id: int = None, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        if user_id is not None:
            # Use database-backed history
            messages = db_manager.get_chat_history(user_id, session_id, limit)
            # Convert to the expected format
            return [
                {
                    "role": msg.role,
                    "content": msg.message,
                    "timestamp": msg.timestamp
                }
                for msg in reversed(messages)  # Reverse to get chronological order
            ]
        else:
            # Fallback to old memory system for backward compatibility
            if hasattr(self, 'chat_sessions'):
                return self.chat_sessions.get_messages(session_id)
            return []
    
    def get_session_context(self, session_id: str) -> tuple:
        """Get the context type and ID for a session"""
        return self.session_manager.get_session_context(session_id)
    
    def validate_session_access(self, session_id: str, user_id: int) -> bool:
        """Validate that a user has access to a session"""
        return self.session_manager.validate_session_access(session_id, user_id)

# Global agent workflow instance
agent_workflow = AgentWorkflow()
