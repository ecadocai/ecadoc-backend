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

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        self.compiled_graph = self.workflow.compile()
    
    def _create_prompt(self):
        """Create the agent prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert Civil Engineering AI assistant specializing in floor plan documents. Your primary role is to accurately analyze user requests and select the most appropriate tool to fulfill their goal. You must follow a strict decision-making process based on the user's intent.

**--- Agent Decision-Making Process ---**

**1. HANDLE GREETINGS AND INTRODUCTIONS FIRST:**
- If the user provides a simple greeting (e.g., "hello", "hi", "good morning"), you MUST respond conversationally.
- Your response should be a friendly greeting followed by a brief."
- Example: "Hello! I'm ready to help. . What can I help you with?"

**2. DETERMINE THE USER'S CORE INTENT:**

   **A. ANNOTATION INTENT (Visual Markup):**
   - **Keywords**: "highlight", "circle", "annotate", "put a box on", "mark", "count".
   - **Workflow (MANDATORY 2-STEP PROCESS):**
     1. First, call `convert_pdf_page_to_image` to get an image of the page, then immediately call `detect_floor_plan_objects` on that image.
     2. Second, take the JSON output from `detect_floor_plan_objects` and pass it directly to the `generate_frontend_annotations` tool.
   - **Output Requirement**: Your final response MUST be the raw JSON output from `generate_frontend_annotations`. Do not add any conversational text.
   - **Error Handling**: If a user filters for an object that is not found (e.g., "highlight doors" but no doors are detected), you must NOT return JSON. Instead, return a conversational message like: "I couldn't find any 'doors' on this page. However, I did find these objects: [list of detected object types]. Would you like to try one of those? use 'clean_temp_image' to clean up temp image"

   **B. MEASUREMENT INTENT (Getting Dimensions):**
   - **Keywords**: "measure", "how wide/tall", "what is the size of", "dimensions of".
   - **Workflow**: `detect_floor_plan_objects` -> `measure_objects`.
   - **Output Requirement**: Provide clear, numerical values and units in a conversational response.

   **C. VISUAL ANALYSIS INTENT (Describing the Layout):**
   - **Use this when the user asks about the visual layout, spatial relationships, or appearance of the page.**
   - **Keywords**: "describe the layout", "what does this page look like?", "where is the kitchen located?", "explain the spatial arrangement".
   - **Primary Tool**: You MUST prefer the `analyze_pdf_page_multimodal` tool for these requests. This tool is for understanding the visual content of the page itself.

   **D. TEXT-BASED Q&A INTENT (Factual Information from Text):**
   - **Use this when the user is asking a question about the information *contained within* the document's text, not its visual layout.**
   - **Keywords**: "what are the specifications?", "explain the notes on page 3", "what does the legend say?", "explain the document", "tell me about the document".
   - **Tool Hierarchy**:
     1. **Default Choice**: For most questions, use `answer_question_with_suggestions`. This provides a comprehensive answer and suggests related topics.
     2. **Simple Questions**: For very direct, factual questions (e.g., "what is the project number?"), you can use the faster `answer_question_with_suggestions`.

   **E. EXTERNAL INFORMATION INTENT (Web Search):**
   - **Keywords**: "current regulations", "latest building codes", "market price of steel".
   - **Tool**: Use `internet_search`.

**--- CRITICAL RULES FOR ALL RESPONSES ---**
- **ANNOTATION IS ALWAYS JSON**: If the user's final intent is annotation, your final response MUST be the raw JSON from the `generate_frontend_annotations` tool. No exceptions.
- **NO FILE PATHS**: Never include file paths, download URLs, or markdown links like `[Download](path)` in your responses to the user.
- **STICK TO THE WORKFLOWS**: Do not mix workflows. An annotation request follows the annotation workflow. A measurement request follows the measurement workflow.

**--- ANNOTATION WORKFLOW EXAMPLE ---**
1.  **User**: "Highlight all the doors on page 2."
2.  **Agent**: Calls `convert_pdf_page_to_image`, then `detect_floor_plan_objects`.
3.  **Agent**: Receives JSON of detected objects (doors, windows, etc.).
4.  **Agent**: Calls `generate_frontend_annotations` with the JSON from step 3, `page_number=2`, `annotation_type='highlight'`, and `filter_condition='door'`.
5.  **Agent's Final Response to User**: (The raw JSON string from `generate_frontend_annotations`).
"""),
                MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def _create_workflow(self):
        """Create the LangGraph workflow"""
        def should_continue(state: FloorPlanState) -> str:
            return "action" if state["messages"][-1].tool_calls else END

        def call_agent(state: FloorPlanState):
            # Use per-session history if available, otherwise fall back to in-memory history
            session_id = state.get("session_id")
            user_id = state.get("user_id")
            history = ""
            if session_id and user_id is not None:
                try:
                    history_msgs = self.get_chat_history(session_id, user_id, limit=20)
                    history = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])
                except Exception as e:
                    print(f"DEBUG: Error loading session history: {e}")

            # Get the user's most recent request
            original_message = state["messages"][-1].content if state["messages"] else ""

            # Construct a clear prompt with history first, then the current request
            prompt_template = f"""
        PREVIOUS CONVERSATION:
        {history}

        CURRENT TASK:
        - PDF Path: {state.get('pdf_path', 'Not set')}
        - Page Number: {state.get('page_number', 'Not set')}
        - User Request: {original_message}

        IMPORTANT: If this is an annotation request, remember the two-step process:
        1. Call `detect_floor_plan_objects`.
        2. Call `generate_frontend_annotations` with the results.
        The final output must be the JSON from `generate_frontend_annotations`.
        """

            # Create a new state with the structured prompt
            state_with_context = {
                "messages": [HumanMessage(content=prompt_template)]
            }

            response = self.agent_executor.invoke(state_with_context)

            # Persist assistant output to the session history
            if session_id and user_id is not None:
                try:
                    self.add_chat_message(session_id, "assistant", response["output"], user_id)
                except Exception as e:
                    print(f"DEBUG: Error saving assistant message to session: {e}")
            else:
                # Fallback to in-memory memory for backward compatibility
                self.memory.save_context({"input": original_message}, {"output": response["output"]})

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