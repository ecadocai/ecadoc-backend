"""
Unified agent API endpoints for the Floor Plan Agent API
"""
import os
import uuid
import re
import json
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import HumanMessage

from modules.config.settings import settings
from modules.config.utils import delete_file_after_delay
from modules.pdf_processing.service import pdf_processor
from modules.database import db_manager
from modules.agent import agent_workflow
from modules.agent.tools import process_question_with_hybrid_search
from modules.projects.service import project_service
from modules.session import session_manager, context_resolver
from modules.auth.deps import get_current_user_id

def extract_manual_suggestions(text: str) -> list:
    """Extract manually formatted suggestions from the response text."""
    suggestions = []
    
    # Look for numbered list items with patterns like:
    # 1. **Title** (Page X): Description
    # or 1. **Title** (Page X) - Description
    pattern = r'(\d+)\. \*\*([^*]+)\*\* \(Page (\d+)\)[:\-] ([^\n]+)'
    
    matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
    
    for match in matches:
        number, title, page, description = match
        try:
            suggestions.append({
                "title": title.strip(),
                "page": int(page),
                "description": description.strip()
            })
        except ValueError:
            continue  # Skip if page number is not valid
    
    print(f"DEBUG: Extracted {len(suggestions)} manual suggestions from text")
    return suggestions


def parse_citations_from_text(text: str, doc_id: str = None) -> tuple:
    """Attempt to parse a simple citation block embedded in plain text responses.

    Looks for a section that starts with 'Document citations:' and collects lines
    that contain page numbers like '[1] Page 3' or 'Page 3'. Returns a tuple of
    (citations_list, most_referenced_page) where citations_list is a list of
    dicts with minimal citation info suitable for the endpoint response.
    """
    citations = []
    most_referenced_page = None

    marker_match = re.search(r"document citations:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if not marker_match:
        # Fallback: scan for lines that include 'Page <num>' anywhere in the text
        lines = text.splitlines()
        page_counts = {}
        for line in lines:
            m = re.search(r"page\s*(\d{1,4})", line, re.IGNORECASE)
            if m:
                page = int(m.group(1))
                idx = len(citations) + 1
                citations.append({
                    "id": idx,
                    "page": page,
                    "text": line.strip()[:500],
                    "relevance_score": None,
                    "doc_id": doc_id
                })
                page_counts[page] = page_counts.get(page, 0) + 1
        if page_counts:
            most_referenced_page = max(page_counts.items(), key=lambda x: x[1])[0]
        return citations, most_referenced_page

    block = marker_match.group(1)
    # Only consider the first 20 lines after the marker for speed
    lines = block.strip().splitlines()[:20]
    page_counts = {}
    for i, line in enumerate(lines):
        # Match patterns like: [1] Page 3, Page 3: some text, 1) Page 3, or just 'Page 3'
        m = re.search(r"\[?(\d+)\]?\s*[:,\)]?\s*Page\s*(\d{1,4})", line, re.IGNORECASE)
        if m:
            try:
                cit_id = int(m.group(1))
            except Exception:
                cit_id = i + 1
            page = int(m.group(2))
        else:
            m2 = re.search(r"Page\s*(\d{1,4})", line, re.IGNORECASE)
            if m2:
                cit_id = i + 1
                page = int(m2.group(1))
            else:
                # No page found on this line, skip
                continue

        citations.append({
            "id": cit_id,
            "page": page,
            "text": line.strip()[:500],
            "relevance_score": None,
            "doc_id": doc_id
        })
        page_counts[page] = page_counts.get(page, 0) + 1

    if page_counts:
        most_referenced_page = max(page_counts.items(), key=lambda x: x[1])[0]

    return citations, most_referenced_page

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post("/unified")
async def unified_agent(
    background_tasks: BackgroundTasks,
    doc_id: str = Form(...),
    user_instruction: str = Form(...),
    user_id: int = Depends(get_current_user_id),
    session_id: str = Form(None)
):
    """
    Single unified endpoint that intelligently handles both chat and annotation workflows.
    The agent automatically determines intent and extracts page information from the instruction.
    """
    # Verify document exists in database
    try:
        doc_info = pdf_processor.get_document_info(doc_id)
    except FileNotFoundError:
        raise HTTPException(404, detail="Document not found")
    
    # Handle different storage types
    pdf_path = None
    if doc_info.get("storage_type") == "database":
        # For database storage, we need to get the PDF content from the database
        try:
            pdf_content = pdf_processor.get_document_content(doc_id)
            # Create a temporary file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                pdf_path = temp_file.name
        except Exception as e:
            raise HTTPException(500, detail=f"Failed to retrieve document content: {str(e)}")
    else:
        # For legacy file storage
        pdf_path = doc_info.get("pdf_path")
        if not pdf_path:
            raise HTTPException(404, detail="Document file path not found")
    
    # Extract page number from instruction or default to 1
    page_number = 1
    page_match = re.search(r'page\s+(\d+)', user_instruction.lower())
    if page_match:
        page_number = int(page_match.group(1))
    
    # Resolve context for session management
    context_data = {'doc_id': doc_id}
    context_type, context_id = context_resolver.resolve_context(context_data)
    
    # Validate context access
    context_resolver.validate_context_access_with_exception(user_id, context_type, context_id)
    
    # Get or create context-aware session
    if session_id:
        # Validate existing session access
        if not session_manager.validate_session_access(session_id, user_id):
            # Create new session if access validation fails
            session_id = session_manager.get_or_create_session(user_id, context_type, context_id)
        else:
            # Update activity for existing session
            session_manager.update_session_activity(session_id)
    else:
        # Create new session with context
        session_id = session_manager.get_or_create_session(user_id, context_type, context_id)
    
    # Add user message to chat history with context
    session_manager.add_message_to_session(session_id, user_id, "user", user_instruction)
    
    # Simple instruction for the agent - let the workflow handle tool selection
    simple_instruction = f"""
Document ID: {doc_id}
Page: {page_number}
User Request: {user_instruction}

Please handle this request using the most appropriate tool.
"""
    
    initial_state = {
        "messages": [HumanMessage(content=simple_instruction)],
        "pdf_path": pdf_path,
        "page_number": page_number,
        "session_id": session_id,
        "user_id": user_id,
    }
    
    try:
        print(f"DEBUG: Starting unified agent for doc {doc_id} with instruction: {user_instruction}")
        final_state = agent_workflow.process_request(initial_state)
        final_msg = final_state["messages"][-1].content
        
        # Save assistant response to chat history
        session_manager.add_message_to_session(session_id, user_id, "assistant", final_msg)

        # Handle the agent's response
        try:
            # Attempt to parse the agent's entire output as JSON
            parsed_data = json.loads(final_msg)
            
            # Case 1: It's the new annotation format
            if isinstance(parsed_data, dict) and 'annotations' in parsed_data:
                print("DEBUG: Annotation JSON response detected.")
                response_data = {
                    "response": parsed_data.get("message", "Annotations generated successfully."),
                    "session_id": session_id,
                    "doc_id": doc_id,
                    "page": page_number,
                    "type": "annotation",
                    "annotation_status": "completed",
                    "annotations": parsed_data.get("annotations", []),
                    "annotation_count": len(parsed_data.get("annotations", [])),
                    "detected_objects": parsed_data.get("detected_objects", []),
                    "message": parsed_data.get("message", "Generated annotations successfully")
                }
                if 'coordinate_space' in parsed_data:
                    response_data['coordinate_space'] = parsed_data['coordinate_space']
                return JSONResponse(content=response_data)
            
            # Case 2: It's a RAG/informational JSON format
            elif isinstance(parsed_data, dict) and 'answer' in parsed_data:
                print("DEBUG: RAG JSON response detected.")
                response_content = {
                    "response": parsed_data.get("answer", "No answer found."),
                    "session_id": session_id,
                    "doc_id": doc_id,
                    "page": page_number,
                    "type": "information",
                    "suggestions": parsed_data.get("suggestions", []),
                    "citations": parsed_data.get("citations", []),
                    "most_referenced_page": parsed_data.get("most_referenced_page")
                }
                # If the agent returned no citations, try to fetch them separately
                try:
                    if not response_content["citations"]:
                        print("DEBUG: No citations found in agent response, attempting to generate citations separately.")
                        hybrid = process_question_with_hybrid_search(doc_id, user_instruction)
                        if hybrid and isinstance(hybrid, dict):
                            response_content["citations"] = hybrid.get("citations", [])
                            # Only override most_referenced_page if we found one
                            if hybrid.get("most_referenced_page"):
                                response_content["most_referenced_page"] = hybrid.get("most_referenced_page")
                except Exception as e:
                    print(f"DEBUG: Failed to generate fallback citations: {e}")
                return JSONResponse(content=response_content)

            # Case 3: It's some other JSON, treat as informational
            else:
                raise json.JSONDecodeError("Not a recognized JSON format", final_msg, 0)

        except json.JSONDecodeError:
            # Case 4: The response is not JSON, so it's a plain text informational response.
            print("DEBUG: Non-JSON response detected. Treating as informational text.")
            answer_text = final_msg
            suggestions = extract_manual_suggestions(answer_text)
            
            if suggestions:
                split_patterns = [r'\n\nHere are some related topics.*', r'\n\nRelated topics.*', r'\n\n\d+\. \*\*.*']
                for pattern in split_patterns:
                    match = re.search(pattern, answer_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        answer_text = answer_text[:match.start()].strip()
                        break
            
            response_content = {
                "response": answer_text,
                "session_id": session_id,
                "doc_id": doc_id,
                "page": page_number,
                "type": "information",
                "suggestions": suggestions,
                "citations": [],
                "most_referenced_page": None
            }
            # Try to extract citations embedded in the plain text answer
            try:
                parsed_citations, parsed_most = parse_citations_from_text(answer_text, doc_id=doc_id)
                if parsed_citations:
                    response_content["citations"] = parsed_citations
                    response_content["most_referenced_page"] = parsed_most
            except Exception as e:
                print(f"DEBUG: Failed to parse inline citations: {e}")
            return JSONResponse(content=response_content)
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"DEBUG: Exception occurred in unified agent: {str(e)}")
        session_manager.add_message_to_session(session_id, user_id, "assistant", error_msg)
        return JSONResponse(
            content={"response": error_msg, "session_id": session_id, "doc_id": doc_id, "type": "error"},
            status_code=500
        )
    finally:
        # Clean up temporary PDF file if it was created
        if doc_info.get("storage_type") == "database" and pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"DEBUG: Cleaned up temporary PDF file: {pdf_path}")
            except Exception as e:
                print(f"DEBUG: Error cleaning up temporary PDF file: {e}")

@router.get("/chat/history")
async def get_chat_history(user_id: int = Depends(get_current_user_id), session_id: str = None, limit: int = 50):
    """
    Retrieve chat history for a specific user.
    If session_id is provided, returns only that session's history.
    Otherwise, returns all chat history for the user.
    """
    try:
        history = db_manager.get_chat_history(user_id, session_id, limit)
        
        # Format the response
        formatted_history = []
        for msg in history:
            formatted_history.append({
                "id": msg.id,
                "session_id": msg.session_id,
                "role": msg.role,
                "message": msg.message,
                "timestamp": msg.timestamp
            })
        
        return {"history": formatted_history}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/chat/sessions")
async def get_user_sessions(user_id: int = Depends(get_current_user_id)):
    """
    Get all unique session IDs for a user.
    """
    try:
        sessions = db_manager.get_user_sessions(user_id)
        return {"sessions": sessions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/project/{project_id}/unified")
async def unified_agent_for_project(
    background_tasks: BackgroundTasks,
    project_id: str,
    user_instruction: str = Form(...),
    user_id: int = Depends(get_current_user_id),
    session_id: str = Form(None),
    doc_id: str = Form(None)
):
    """
    Project-aware unified endpoint that works with project context.
    Automatically extracts the document from the project and provides project context.
    """
    # Validate project access
    if not project_service.validate_project_access(project_id, user_id):
        raise HTTPException(403, detail="Access denied or project not found")
    
    # Get project information
    project = project_service.get_project(project_id)
    if not project or not project.get("documents") or len(project["documents"]) == 0:
        raise HTTPException(400, detail="Project has no associated document")
    
    # Select the document to use
    selected_document = None
    if doc_id:
        for doc in project["documents"]:
            if doc["doc_id"] == doc_id:
                selected_document = doc
                break
        if not selected_document:
            available_docs = [doc["doc_id"] for doc in project["documents"]]
            raise HTTPException(400, detail=f"Document {doc_id} not found in project. Available documents: {available_docs}")
    else:
        selected_document = project["documents"][0]
    
    final_doc_id = selected_document["doc_id"]
    
    # Verify document exists in database
    try:
        doc_info = pdf_processor.get_document_info(final_doc_id)
    except FileNotFoundError:
        raise HTTPException(404, detail="Document not found")
    
    # Handle different storage types
    pdf_path = None
    if doc_info.get("storage_type") == "database":
        try:
            pdf_content = pdf_processor.get_document_content(final_doc_id)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                pdf_path = temp_file.name
        except Exception as e:
            raise HTTPException(500, detail=f"Failed to retrieve document content: {str(e)}")
    else:
        pdf_path = doc_info.get("pdf_path")
        if not pdf_path:
            raise HTTPException(404, detail="Document file path not found")
    
    # Extract page number
    page_number = 1
    page_match = re.search(r'page\s+(\d+)', user_instruction.lower())
    if page_match:
        page_number = int(page_match.group(1))
    
    # Resolve and validate context
    context_data = {'project_id': project_id, 'doc_id': final_doc_id}
    context_type, context_id = context_resolver.resolve_context(context_data)
    context_resolver.validate_context_access_with_exception(user_id, context_type, context_id)
    
    # Get or create session
    if session_id:
        if not session_manager.validate_session_access(session_id, user_id):
            session_id = session_manager.get_or_create_session(user_id, context_type, context_id)
        else:
            session_manager.update_session_activity(session_id)
    else:
        session_id = session_manager.get_or_create_session(user_id, context_type, context_id)
    
    # Add message to history
    session_manager.add_message_to_session(session_id, user_id, "user", user_instruction)
    
    # Prepare agent instruction
    simple_instruction = f"""
Project Context:
- Project ID: {project_id}
- Project Name: {project["name"]}
- Document ID: {final_doc_id}
- Document: {selected_document["filename"]}
- Page: {page_number}

User Request: {user_instruction}

Please handle this request using the most appropriate tool.
"""
    
    initial_state = {
        "messages": [HumanMessage(content=simple_instruction)],
        "pdf_path": pdf_path,
        "page_number": page_number,
        "session_id": session_id,
        "user_id": user_id,
    }
    
    try:
        print(f"DEBUG: Starting project agent for project {project_id}, doc {final_doc_id}")
        final_state = agent_workflow.process_request(initial_state)
        final_msg = final_state["messages"][-1].content
        
        # Save assistant response
        session_manager.add_message_to_session(session_id, user_id, "assistant", final_msg)
        
        # Handle agent response
        try:
            parsed_data = json.loads(final_msg)
            
            # Case 1: Annotation JSON
            if isinstance(parsed_data, dict) and 'annotations' in parsed_data:
                print("DEBUG: Project annotation JSON response detected.")
                response_data = {
                    "response": parsed_data.get("message", "Annotations generated successfully."),
                    "session_id": session_id,
                    "project_id": project_id,
                    "doc_id": final_doc_id,
                    "page": page_number,
                    "type": "annotation",
                    "annotation_status": "completed",
                    "annotations": parsed_data.get("annotations", []),
                    "annotation_count": len(parsed_data.get("annotations", [])),
                    "detected_objects": parsed_data.get("detected_objects", []),
                    "message": parsed_data.get("message", "Generated annotations successfully"),
                    "project_context": {"name": project["name"], "description": project["description"]}
                }
                if 'coordinate_space' in parsed_data:
                    response_data['coordinate_space'] = parsed_data['coordinate_space']
                return JSONResponse(content=response_data)
            
            # Case 2: RAG/informational JSON
            elif isinstance(parsed_data, dict) and 'answer' in parsed_data:
                print("DEBUG: Project RAG JSON response detected.")
                response_content = {
                    "response": parsed_data.get("answer", "No answer found."),
                    "session_id": session_id,
                    "project_id": project_id,
                    "doc_id": final_doc_id,
                    "page": page_number,
                    "type": "information",
                    "suggestions": parsed_data.get("suggestions", []),
                    "citations": parsed_data.get("citations", []),
                    "most_referenced_page": parsed_data.get("most_referenced_page"),
                    "project_context": {"name": project["name"], "description": project["description"]}
                }
                # If the agent returned no citations, attempt to fetch them separately
                try:
                    if not response_content["citations"]:
                        print("DEBUG: No citations found in project agent response, attempting to generate citations separately.")
                        hybrid = process_question_with_hybrid_search(final_doc_id, user_instruction)
                        if hybrid and isinstance(hybrid, dict):
                            response_content["citations"] = hybrid.get("citations", [])
                            if hybrid.get("most_referenced_page"):
                                response_content["most_referenced_page"] = hybrid.get("most_referenced_page")
                except Exception as e:
                    print(f"DEBUG: Failed to generate fallback project citations: {e}")
                return JSONResponse(content=response_content)
            
            else:
                raise json.JSONDecodeError("Not a recognized JSON format", final_msg, 0)
        
        except json.JSONDecodeError:
            # Case 3: Plain text response
            print("DEBUG: Project non-JSON response detected.")
            answer_text = final_msg
            suggestions = extract_manual_suggestions(answer_text)
            
            if suggestions:
                split_patterns = [r'\n\nHere are some related topics.*', r'\n\nRelated topics.*', r'\n\n\d+\. \*\*.*']
                for pattern in split_patterns:
                    match = re.search(pattern, answer_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        answer_text = answer_text[:match.start()].strip()
                        break
            
            response_content = {
                "response": answer_text,
                "session_id": session_id,
                "project_id": project_id,
                "doc_id": final_doc_id,
                "page": page_number,
                "type": "information",
                "suggestions": suggestions,
                "citations": [],
                "most_referenced_page": None,
                "project_context": {"name": project["name"], "description": project["description"]}
            }
            # Try to extract citations embedded in the plain text answer
            try:
                parsed_citations, parsed_most = parse_citations_from_text(answer_text, doc_id=final_doc_id)
                if parsed_citations:
                    response_content["citations"] = parsed_citations
                    response_content["most_referenced_page"] = parsed_most
            except Exception as e:
                print(f"DEBUG: Failed to parse inline citations in project response: {e}")
            return JSONResponse(content=response_content)
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"DEBUG: Exception occurred in project agent: {str(e)}")
        session_manager.add_message_to_session(session_id, user_id, "assistant", error_msg)
        return JSONResponse(
            content={"response": error_msg, "session_id": session_id, "project_id": project_id, "type": "error"},
            status_code=500
        )
    finally:
        # Clean up temporary PDF file if it was created
        if doc_info.get("storage_type") == "database" and pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"DEBUG: Cleaned up temporary PDF file: {pdf_path}")
            except Exception as e:
                print(f"DEBUG: Error cleaning up temporary PDF file: {e}")