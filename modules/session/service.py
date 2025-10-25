"""
Session management service for the Floor Plan Agent API
"""
import uuid
import random
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from modules.config.settings import settings
from modules.database.models import db_manager, ChatSession
from .exceptions import (
    SessionError, SessionNotFoundError, SessionAccessDeniedError,
    InvalidContextError, SessionExpiredError, SessionCreationError
)

class SessionManager:
    """Enhanced session management service with context support"""
    
    def __init__(self):
        self.db = db_manager
        # In-memory cache for active sessions to improve performance
        self._session_cache = {}
        self._cache_max_size = settings.SESSION_CACHE_MAX_SIZE
    
    def create_session(self, user_id: int, context_type: str, context_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Create a new session with context support"""
        session_id = str(uuid.uuid4())
        
        # Validate context type
        if context_type not in ['PROJECT', 'DOCUMENT', 'GENERAL']:
            raise InvalidContextError(f"Invalid context_type: {context_type}. Must be PROJECT, DOCUMENT, or GENERAL")
        
        # Create session in database
        success = self.db.create_chat_session(
            session_id=session_id,
            user_id=user_id,
            context_type=context_type,
            context_id=context_id,
            metadata=metadata
        )
        
        if not success:
            raise SessionCreationError("Failed to create session in database")
        
        # Add to cache
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            context_type=context_type,
            context_id=context_id,
            is_active=True,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata=metadata
        )
        self._add_to_cache(session_id, session)
        
        return session_id
    
    def get_or_create_session(self, user_id: int, context_type: str, context_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Get existing session or create new one for the given context"""
        # Try to find existing active session for this context
        existing_session = self.db.get_session_by_context(user_id, context_type, context_id)
        
        if existing_session and existing_session.is_active:
            # Update activity and return existing session
            self.update_session_activity(existing_session.session_id)
            self._add_to_cache(existing_session.session_id, existing_session)
            return existing_session.session_id
        
        # Create new session
        return self.create_session(user_id, context_type, context_id, metadata)
    
    def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        """Get session by session ID with caching"""
        # Check cache first
        if session_id in self._session_cache:
            return self._session_cache[session_id]
        
        # Get from database
        session = self.db.get_session_by_id(session_id)
        if session:
            self._add_to_cache(session_id, session)
        
        return session
    
    def get_active_sessions(self, user_id: int, context_type: str = None) -> List[ChatSession]:
        """Get all active sessions for a user, optionally filtered by context type"""
        return self.db.get_active_sessions(user_id, context_type)
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session activity timestamp"""
        success = self.db.update_session_activity(session_id)
        
        # Update cache if session exists there
        if session_id in self._session_cache:
            self._session_cache[session_id].last_activity = datetime.now()
        
        # Randomly trigger cleanup
        if random.random() < settings.SESSION_CLEANUP_PROBABILITY:
            self.cleanup_expired_sessions()
        
        return success
    
    def deactivate_session(self, session_id: str) -> bool:
        """Mark session as inactive"""
        success = self.db.deactivate_session(session_id)
        
        # Update cache
        if session_id in self._session_cache:
            self._session_cache[session_id].is_active = False
        
        return success
    
    def cleanup_expired_sessions(self, hours: int = None) -> int:
        """Clean up expired sessions"""
        if hours is None:
            hours = settings.SESSION_CLEANUP_HOURS
        
        cleaned_count = self.db.cleanup_expired_sessions(hours)
        
        # Clear cache of expired sessions
        self._clear_expired_cache(hours)
        
        return cleaned_count
    
    def add_message_to_session(self, session_id: str, user_id: int, role: str, message: str) -> bool:
        """Add a message to a session with automatic context detection"""
        # Get session to determine context
        session = self.get_session_by_id(session_id)
        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        # Validate session access
        if session.user_id != user_id:
            raise SessionAccessDeniedError(f"User {user_id} does not have access to session {session_id}")
        
        # Check if session is active
        if not session.is_active:
            raise SessionExpiredError(f"Session {session_id} is not active")
        
        # Add message with context information
        self.db.add_chat_message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            message=message,
            context_type=session.context_type,
            context_id=session.context_id
        )
        
        return True
    
    def get_session_context(self, session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Get the context type and context ID for a session"""
        session = self.get_session_by_id(session_id)
        if session:
            return session.context_type, session.context_id
        return None, None

    def get_session_by_context(self, user_id: int, context_type: str, context_id: str = None) -> Optional[str]:
        """Return the session_id for the most recent active session matching the context for the user.

        This performs a read-only lookup and will NOT create a session if none exists.
        """
        session = self.db.get_session_by_context(user_id, context_type, context_id)
        return session.session_id if session else None
    
    def validate_session_access(self, session_id: str, user_id: int) -> bool:
        """Validate that a user has access to a session"""
        session = self.get_session_by_id(session_id)
        return session is not None and session.user_id == user_id
    
    def get_sessions_by_context(self, user_id: int, context_type: str, context_id: str = None) -> List[ChatSession]:
        """Get all sessions (active and inactive) for a specific context"""
        # This would require a new database method, for now return active sessions
        return self.get_active_sessions(user_id, context_type)
    
    def _add_to_cache(self, session_id: str, session: ChatSession):
        """Add session to cache with size management"""
        # If cache is full, remove oldest entries
        if len(self._session_cache) >= self._cache_max_size:
            # Remove 10% of oldest entries
            remove_count = max(1, self._cache_max_size // 10)
            oldest_keys = list(self._session_cache.keys())[:remove_count]
            for key in oldest_keys:
                del self._session_cache[key]
        
        self._session_cache[session_id] = session
    
    def _clear_expired_cache(self, hours: int):
        """Clear expired sessions from cache"""
        now = datetime.now()
        expired_keys = []
        
        for session_id, session in self._session_cache.items():
            if session.last_activity:
                time_diff = (now - session.last_activity).total_seconds() / 3600
                if time_diff > hours:
                    expired_keys.append(session_id)
        
        for key in expired_keys:
            del self._session_cache[key]
    
    def clear_cache(self):
        """Clear the entire session cache"""
        self._session_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cache_size": len(self._session_cache),
            "cache_max_size": self._cache_max_size,
            "cache_usage_percent": (len(self._session_cache) / self._cache_max_size) * 100
        }

# Global session manager instance
session_manager = SessionManager()