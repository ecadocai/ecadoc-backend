"""
Unified agent API endpoints for the Floor Plan Agent API
"""
import os
import asyncio
import threading
import queue
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
from modules.agent.progress import set_progress_callback

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
                    "text": f"Page {page}",
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
            "text": f"Page {page}",
            "relevance_score": None,
            "doc_id": doc_id
        })
        page_counts[page] = page_counts.get(page, 0) + 1

    if page_counts:
        most_referenced_page = max(page_counts.items(), key=lambda x: x[1])[0]

    return citations, most_referenced_page


def _ensure_citations(doc_id: str, question: str, answer_text, page_number: int):
    """Best-effort citation generator that guarantees a non-empty list.

    Order of attempts:
    1) Hybrid search (preferred)
    2) Parse inline citations from text (fallback)
    3) Minimal placeholder citation pointing to the working page
    """
    try:
        hybrid = process_question_with_hybrid_search(doc_id, question)
        if hybrid and isinstance(hybrid, dict):
            cits = hybrid.get("citations") or []
            # Dedupe by text+page and cap
            seen = set()
            unique = []
            for c in cits:
                key = f"{c.get('text')}:{c.get('page')}"
                if key in seen:
                    continue
                seen.add(key)
                unique.append(c)
            mrp = hybrid.get("most_referenced_page")
            if unique:
                return unique[:5], mrp
    except Exception as e:
        print(f"DEBUG: ensure_citations hybrid search failed: {e}")

    try:
        if answer_text:
            parsed_citations, parsed_most = parse_citations_from_text(answer_text, doc_id=doc_id)
            if parsed_citations:
                return parsed_citations, parsed_most
    except Exception as e:
        print(f"DEBUG: ensure_citations parse failed: {e}")

    # Final guaranteed placeholder
    placeholder = [{
        "id": 1,
        "page": int(page_number) if page_number else 1,
        "text": f"Page {int(page_number) if page_number else 1}",
        "relevance_score": None,
        "doc_id": doc_id,
    }]
    return placeholder, placeholder[0]["page"]


def _filter_citations_to_best_page(citations: list, target_page: int = None, most_referenced_page: int = None):
    """Dedupe citations and return only those that belong to the best page.

    Best page priority: target_page (working page) > most_referenced_page > majority vote.
    Returns (filtered_citations, best_page).
    """
    if not citations:
        return [], most_referenced_page or target_page

    # Dedupe by (text,page)
    seen = set()
    unique = []
    for c in citations:
        key = f"{c.get('text')}:{c.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    # Decide best page
    best = None
    if isinstance(target_page, int) and target_page > 0:
        best = target_page
    elif isinstance(most_referenced_page, int) and most_referenced_page > 0:
        best = most_referenced_page
    else:
        counts = {}
        for c in unique:
            p = c.get('page')
            counts[p] = counts.get(p, 0) + 1
        if counts:
            best = max(counts.items(), key=lambda x: x[1])[0]

    # Filter to best page, and normalize label
    filtered = [c for c in unique if c.get('page') == best]
    if not filtered:
        filtered = unique[:1]

    compact = []
    seen_pages = set()
    for c in filtered:
        p = c.get('page', 1)
        if p in seen_pages:
            continue
        seen_pages.add(p)
        compact.append({
            'id': len(compact) + 1,
            'page': p,
            'text': f'Page {p}',
            'relevance_score': None,
            'doc_id': c.get('doc_id')
        })
    return compact, (best or (compact[0]['page'] if compact else None))


def _build_persisted_message(answer_text: str, citations: list) -> str:
    """Append a compact, parseable citations JSON marker to the assistant message.

    The client can parse and render citations from history while hiding the marker.
    Format: <<CITATIONS:{json}>>
    """
    try:
        minimal = [
            {"page": int(c.get("page", 1)), "text": f"Page {int(c.get('page', 1))}"}
            for c in citations or []
        ]
        marker = f"\n\n<<CITATIONS:{json.dumps(minimal)}>>"
        return (answer_text or "").rstrip() + marker
    except Exception:
        return answer_text or ""

router = APIRouter(prefix="/agent", tags=["agent"])

def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string with JSON data."""
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        payload = json.dumps({"message": str(data)})
    return f"event: {event}\ndata: {payload}\n\n"

def _format_status_message(code: str, data: dict) -> str:
    """Map progress codes to concise, human-friendly messages."""
    page = data.get("page")
    count = data.get("count")
    if code == "pdf_load_start":
        return "Loading PDF…"
    if code == "pdf_load_complete":
        return "PDF ready"
    if code == "page_image_start":
        return f"Rasterizing page {page}…" if page else "Rasterizing page…"
    if code == "page_image_ready":
        return f"Page {page} image ready" if page else "Page image ready"
    if code == "object_detection_start":
        return "Detecting objects…"
    if code == "object_detection_complete":
        return f"Detected {count} objects" if count is not None else "Detection complete"
    if code == "annotations_generate_start":
        return "Generating annotations…"
    if code == "annotations_generate_complete":
        return f"Applied {count} annotation(s)" if count is not None else "Annotations ready"
    if code == "vision_analysis_start":
        return f"Analyzing page {page} visually…" if page else "Analyzing page visually…"
    if code == "vision_analysis_complete":
        return "Visual analysis complete"
    if code == "rag_start":
        return "Retrieving relevant text…"
    if code == "rag_complete":
        return "Formulating answer…"
    if code == "rag_suggestions_start":
        return "Finding related topics…"
    if code == "rag_suggestions_complete":
        return "Suggestions ready"
    # Errors and warnings
    if code.endswith("_error"):
        return (data.get("message") or code.replace("_", " ").title())
    if code.endswith("_warning"):
        return (data.get("message") or code.replace("_", " ").title())
    return code.replace("_", " ").title()

async def _unified_stream_impl(
    *,
    background_tasks: BackgroundTasks,
    user_id: int,
    doc_id: str,
    user_instruction: str,
    session_id: str | None,
    project_id: str | None = None,
):
    """Async generator that yields SSE status updates and a final payload.

    Mirrors the logic of the non-streaming unified endpoints but sends
    coarse, non-sensitive progress events so the client can display
    accurate statuses. The final emitted event is `final` with the
    same JSON schema as the regular endpoint.
    """
    # 1) Validate and resolve document
    yield _sse_event("status", {"stage": "accepted", "message": "Request accepted"})
    try:
        doc_info = pdf_processor.get_document_info(doc_id)
    except FileNotFoundError:
        yield _sse_event("error", {"message": "Document not found"})
        yield _sse_event("done", {"ok": False})
        return

    yield _sse_event("status", {"stage": "resolving_context", "message": "Resolving access and session"})

    # Handle different storage types
    pdf_path = None
    if doc_info.get("storage_type") == "database":
        try:
            pdf_content = pdf_processor.get_document_content(doc_id)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                pdf_path = temp_file.name
        except Exception as e:
            yield _sse_event("error", {"message": f"Failed to retrieve document content: {str(e)}"})
            yield _sse_event("done", {"ok": False})
            return
    else:
        pdf_path = doc_info.get("pdf_path")
        if not pdf_path:
            yield _sse_event("error", {"message": "Document file path not found"})
            yield _sse_event("done", {"ok": False})
            return

    try:
        if doc_info.get("storage_type") == "database" and pdf_path:
            background_tasks.add_task(delete_file_after_delay, pdf_path, 30)
    except Exception:
        # never fail due to cleanup scheduling
        pass

    # 2) Page selection / quick retrieval
    page_number = 1
    page_match = re.search(r'page\s+(\d+)', user_instruction.lower())
    if page_match:
        page_number = int(page_match.group(1))

    prefetched_citations = None
    if not page_match:
        yield _sse_event("status", {"stage": "selecting_page", "message": "Selecting best page", "hint": "quick_hybrid_search"})
        try:
            quick = process_question_with_hybrid_search(doc_id, user_instruction)
            if isinstance(quick, dict):
                best = quick.get("most_referenced_page")
                if isinstance(best, int) and best > 0:
                    page_number = best
                prefetched_citations = quick.get("citations") or None
        except Exception as e:
            # Non-fatal; continue with default page
            yield _sse_event("status", {"stage": "selecting_page", "message": f"Page selection fallback: {str(e)}"})

    # 3) Resolve session context
    context_data = {"doc_id": doc_id}
    context_type, context_id = context_resolver.resolve_context(context_data)
    context_resolver.validate_context_access_with_exception(user_id, context_type, context_id)
    if session_id:
        if not session_manager.validate_session_access(session_id, user_id):
            session_id = session_manager.get_or_create_session(user_id, context_type, context_id)
        else:
            session_manager.update_session_activity(session_id)
    else:
        session_id = session_manager.get_or_create_session(user_id, context_type, context_id)

    # Persist user message
    session_manager.add_message_to_session(session_id, user_id, "user", user_instruction)

    # 4) Build request and run agent
    yield _sse_event("status", {"stage": "agent_start", "message": "Invoking agent"})
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

    # Install progress callback and run agent in a worker thread so tool-level
    # events can be forwarded in real time via SSE.
    q: "queue.Queue[dict]" = queue.Queue()
    def progress_cb(evt: dict):
        try:
            q.put(evt, block=False)
        except Exception:
            pass
    final_state_holder = {"state": None, "error": None}
    def _run_agent():
        try:
            # Attach progress callback in the worker thread context
            set_progress_callback(progress_cb)
            st = agent_workflow.process_request(initial_state)
            final_state_holder["state"] = st
        except Exception as ex:
            final_state_holder["error"] = ex

    t = threading.Thread(target=_run_agent, daemon=True)
    t.start()

    # Drain status queue while agent runs
    while t.is_alive():
        try:
            evt = q.get(timeout=0.25)
            msg = _format_status_message(evt.get("code", "status"), evt)
            yield _sse_event("status", {"stage": evt.get("code", "status"), "message": msg, **evt})
        except queue.Empty:
            await asyncio.sleep(0.1)

    # Flush remaining events, if any
    while not q.empty():
        try:
            evt = q.get_nowait()
            msg = _format_status_message(evt.get("code", "status"), evt)
            yield _sse_event("status", {"stage": evt.get("code", "status"), "message": msg, **evt})
        except Exception:
            break

    # Clear callback for this request (best-effort)
    try:
        set_progress_callback(None)
    except Exception:
        pass

    if final_state_holder["error"] is not None:
        e = final_state_holder["error"]
        yield _sse_event("error", {"message": f"Agent error: {str(e)}"})
        yield _sse_event("done", {"ok": False})
        return

    final_state = final_state_holder["state"]
    final_msg = final_state["messages"][-1].content
    yield _sse_event("status", {"stage": "agent_complete", "message": "Agent completed"})

    # 5) Postprocess into unified response
    yield _sse_event("status", {"stage": "postprocess", "message": "Parsing response"})
    try:
        parsed_data = json.loads(final_msg)
        # Annotation JSON
        if isinstance(parsed_data, dict) and 'annotations' in parsed_data:
            session_manager.add_message_to_session(
                session_id, user_id, "assistant", parsed_data.get("message", "Annotations generated successfully.")
            )
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
            if 'download_url' in parsed_data:
                response_data['download_url'] = parsed_data['download_url']
            if 'annotated_file_id' in parsed_data:
                response_data['annotated_file_id'] = parsed_data['annotated_file_id']
            if 'annotated_file_path' in parsed_data:
                response_data['annotated_file_path'] = parsed_data['annotated_file_path']
            yield _sse_event("final", response_data)
            yield _sse_event("done", {"ok": True})
            return
        # RAG JSON
        elif isinstance(parsed_data, dict) and 'answer' in parsed_data:
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
            if not response_content["citations"]:
                if prefetched_citations:
                    filtered, bestp = _filter_citations_to_best_page(prefetched_citations, page_number, response_content.get("most_referenced_page"))
                    response_content["citations"] = filtered
                    if not response_content.get("most_referenced_page"):
                        response_content["most_referenced_page"] = bestp
                else:
                    cits, mrp = _ensure_citations(doc_id, user_instruction, response_content["response"], page_number)
                    filtered, bestp = _filter_citations_to_best_page(cits, page_number, mrp)
                    response_content["citations"] = filtered
                    if not response_content.get("most_referenced_page"):
                        response_content["most_referenced_page"] = bestp
            persisted = _build_persisted_message(response_content["response"], response_content.get("citations") or [])
            session_manager.add_message_to_session(session_id, user_id, "assistant", persisted)
            yield _sse_event("final", response_content)
            yield _sse_event("done", {"ok": True})
            return
        else:
            raise json.JSONDecodeError("Not a recognized JSON format", final_msg, 0)
    except json.JSONDecodeError:
        # Plain text informational response
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
        if prefetched_citations:
            filtered, bestp = _filter_citations_to_best_page(prefetched_citations, page_number, response_content.get("most_referenced_page"))
            response_content["citations"] = filtered
            if not response_content.get("most_referenced_page"):
                response_content["most_referenced_page"] = bestp
        else:
            cits, mrp = _ensure_citations(doc_id, user_instruction, answer_text, page_number)
            filtered, bestp = _filter_citations_to_best_page(cits, page_number, mrp)
            response_content["citations"] = filtered
            response_content["most_referenced_page"] = bestp
        persisted = _build_persisted_message(response_content["response"], response_content.get("citations") or [])
        session_manager.add_message_to_session(session_id, user_id, "assistant", persisted)
        yield _sse_event("final", response_content)
        yield _sse_event("done", {"ok": True})
        return
    except Exception as e:
        yield _sse_event("error", {"message": f"Error processing request: {str(e)}"})
        yield _sse_event("done", {"ok": False})
    finally:
        # Clean up temporary PDF if needed
        if doc_info.get("storage_type") == "database" and pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass

@router.post("/unified/stream")
async def unified_agent_stream(
    background_tasks: BackgroundTasks,
    doc_id: str = Form(...),
    user_instruction: str = Form(...),
    user_id: int = Depends(get_current_user_id),
    session_id: str = Form(None),
):
    async def event_gen():
        async for chunk in _unified_stream_impl(
            background_tasks=background_tasks,
            user_id=user_id,
            doc_id=doc_id,
            user_instruction=user_instruction,
            session_id=session_id,
            project_id=None,
        ):
            yield chunk

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })

@router.post("/project/{project_id}/unified/stream")
async def project_unified_agent_stream(
    project_id: str,
    background_tasks: BackgroundTasks,
    doc_id: str = Form(None),
    user_instruction: str = Form(...),
    user_id: int = Depends(get_current_user_id),
    session_id: str = Form(None),
):
    """Project-aware streaming endpoint.

    If doc_id is omitted, it attempts to infer the active document from the project context.
    """
    # Infer document id if missing
    resolved_doc_id = doc_id
    if not resolved_doc_id:
        try:
            project = project_service.get_project_by_id(project_id)
            resolved_doc_id = (project or {}).get("doc_id") or (project or {}).get("document_id") or doc_id
        except Exception:
            resolved_doc_id = doc_id
    async def event_gen():
        async for chunk in _unified_stream_impl(
            background_tasks=background_tasks,
            user_id=user_id,
            doc_id=resolved_doc_id,
            user_instruction=user_instruction,
            session_id=session_id,
            project_id=project_id,
        ):
            yield chunk

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })

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
    # Ensure temporary PDF created for DB storage is cleaned up after the response
    try:
        if doc_info.get("storage_type") == "database" and pdf_path:
            # Schedule deletion after response is sent (non-blocking)
            background_tasks.add_task(delete_file_after_delay, pdf_path, 30)
    except Exception:
        # Never fail request due to cleanup scheduling
        pass

    # Extract page number from instruction or default to 1
    page_number = 1
    page_match = re.search(r'page\s+(\d+)', user_instruction.lower())
    if page_match:
        page_number = int(page_match.group(1))

    # If no explicit page was requested, pick the best page using fast retrieval
    prefetched_citations = None
    if not page_match:
        try:
            quick = process_question_with_hybrid_search(doc_id, user_instruction)
            if isinstance(quick, dict):
                best = quick.get("most_referenced_page")
                if isinstance(best, int) and best > 0:
                    page_number = best
                prefetched_citations = quick.get("citations") or None
        except Exception as e:
            print(f"DEBUG: quick page selection failed: {e}")
    
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

        # Handle the agent's response
        try:
            # Attempt to parse the agent's entire output as JSON
            parsed_data = json.loads(final_msg)
            
            # Case 1: It's the new annotation format
            if isinstance(parsed_data, dict) and 'annotations' in parsed_data:
                print("DEBUG: Annotation JSON response detected.")
                # Save only the message text to history
                session_manager.add_message_to_session(
                    session_id, user_id, "assistant", parsed_data.get("message", "Annotations generated successfully.")
                )
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
                # Prefer prefetched citations, otherwise ensure; then filter to best page
                if not response_content["citations"]:
                    if prefetched_citations:
                        filtered, bestp = _filter_citations_to_best_page(prefetched_citations, page_number, response_content.get("most_referenced_page"))
                        response_content["citations"] = filtered
                        if not response_content.get("most_referenced_page"):
                            response_content["most_referenced_page"] = bestp
                    else:
                        cits, mrp = _ensure_citations(doc_id, user_instruction, response_content["response"], page_number)
                        filtered, bestp = _filter_citations_to_best_page(cits, page_number, mrp)
                        response_content["citations"] = filtered
                        if not response_content.get("most_referenced_page"):
                            response_content["most_referenced_page"] = bestp
                # Persist message with citations marker
                persisted = _build_persisted_message(response_content["response"], response_content.get("citations") or [])
                session_manager.add_message_to_session(session_id, user_id, "assistant", persisted)
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
            # Ensure citations are present; then filter to best page
            if prefetched_citations:
                filtered, bestp = _filter_citations_to_best_page(prefetched_citations, page_number, response_content.get("most_referenced_page"))
                response_content["citations"] = filtered
                if not response_content.get("most_referenced_page"):
                    response_content["most_referenced_page"] = bestp
            else:
                cits, mrp = _ensure_citations(doc_id, user_instruction, answer_text, page_number)
                filtered, bestp = _filter_citations_to_best_page(cits, page_number, mrp)
                response_content["citations"] = filtered
                response_content["most_referenced_page"] = bestp

            # Persist message with citations marker
            persisted = _build_persisted_message(response_content["response"], response_content.get("citations") or [])
            session_manager.add_message_to_session(session_id, user_id, "assistant", persisted)
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

    # If no page specified, choose best page via quick retrieval
    prefetched_citations = None
    if not page_match:
        try:
            quick = process_question_with_hybrid_search(final_doc_id, user_instruction)
            if isinstance(quick, dict):
                best = quick.get("most_referenced_page")
                if isinstance(best, int) and best > 0:
                    page_number = best
                prefetched_citations = quick.get("citations") or None
        except Exception as e:
            print(f"DEBUG: quick page selection failed (project): {e}")
    
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
        
        # Handle agent response and persist concise message
        try:
            parsed_data = json.loads(final_msg)
            
            # Case 1: Annotation JSON
            if isinstance(parsed_data, dict) and 'annotations' in parsed_data:
                print("DEBUG: Project annotation JSON response detected.")
                # Save only the message text, not full annotation payload
                session_manager.add_message_to_session(
                    session_id, user_id, "assistant", parsed_data.get("message", "Annotations generated successfully.")
                )
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
                # We'll persist after citations are finalized
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
                # Ensure citations are present; then filter to best page
                if not response_content["citations"]:
                    if prefetched_citations:
                        filtered, bestp = _filter_citations_to_best_page(prefetched_citations, page_number, response_content.get("most_referenced_page"))
                        response_content["citations"] = filtered
                        if not response_content.get("most_referenced_page"):
                            response_content["most_referenced_page"] = bestp
                    else:
                        cits, mrp = _ensure_citations(final_doc_id, user_instruction, response_content["response"], page_number)
                        filtered, bestp = _filter_citations_to_best_page(cits, page_number, mrp)
                        response_content["citations"] = filtered
                        if not response_content.get("most_referenced_page"):
                            response_content["most_referenced_page"] = bestp
                # Persist with citations marker
                persisted = _build_persisted_message(response_content["response"], response_content.get("citations") or [])
                session_manager.add_message_to_session(session_id, user_id, "assistant", persisted)
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
            # We'll persist after citations are finalized
            # Ensure citations are present; then filter to best page
            if prefetched_citations:
                filtered, bestp = _filter_citations_to_best_page(prefetched_citations, page_number, response_content.get("most_referenced_page"))
                response_content["citations"] = filtered
                if not response_content.get("most_referenced_page"):
                    response_content["most_referenced_page"] = bestp
            else:
                cits, mrp = _ensure_citations(final_doc_id, user_instruction, answer_text, page_number)
                filtered, bestp = _filter_citations_to_best_page(cits, page_number, mrp)
                response_content["citations"] = filtered
                response_content["most_referenced_page"] = bestp
            persisted = _build_persisted_message(response_content["response"], response_content.get("citations") or [])
            session_manager.add_message_to_session(session_id, user_id, "assistant", persisted)
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
