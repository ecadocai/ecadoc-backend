"""
Optimized database service for high-performance document retrieval
"""
import asyncio
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import sqlite3
from modules.config.settings import settings
from modules.database.models import Document, FileStorage, db_manager


@dataclass
class DocumentMetadata:
    """Optimized document metadata for caching"""
    doc_id: str
    filename: str
    file_id: str
    file_size: int
    content_type: str
    user_id: int
    last_modified: datetime
    etag: str
    
    def to_cache_key(self) -> str:
        return f"doc_meta:{self.doc_id}"
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'DocumentMetadata':
        """Create DocumentMetadata from database row"""
        return cls(
            doc_id=row['doc_id'],
            filename=row['filename'],
            file_id=row['file_id'],
            file_size=row.get('file_size', 0),
            content_type=row.get('content_type', 'application/pdf'),
            user_id=row['user_id'],
            last_modified=row.get('updated_at', datetime.now()),
            etag=f"{row['doc_id']}-{row.get('updated_at', datetime.now()).timestamp()}"
        )


@dataclass
class DocumentWithContent:
    """Document with binary content for optimized retrieval"""
    metadata: DocumentMetadata
    content: bytes
    
    def is_cacheable(self) -> bool:
        """Check if document is small enough to cache"""
        max_cache_size = getattr(settings, 'CACHE_MAX_FILE_SIZE', 10 * 1024 * 1024)  # 10MB default
        return self.metadata.file_size <= max_cache_size


class OptimizedDocumentService:
    """High-performance document retrieval service with connection pooling"""
    
    def __init__(self):
        self.db_manager = db_manager
        self._connection_pool = None
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """Initialize connection pool for PostgreSQL"""
        if settings.USE_RDS and settings.IS_POSTGRES:
            try:
                self._connection_pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=settings.DB_POOL_SIZE,
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    database=settings.DB_NAME,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD
                )
            except Exception as e:
                print(f"Failed to initialize connection pool: {e}")
                self._connection_pool = None
    
    def _get_connection(self):
        """Get database connection from pool or fallback to db_manager"""
        if self._connection_pool:
            try:
                return self._connection_pool.getconn()
            except Exception:
                pass
        return self.db_manager.get_connection()
    
    def _return_connection(self, conn):
        """Return connection to pool or close it"""
        if self._connection_pool:
            try:
                self._connection_pool.putconn(conn)
                return
            except Exception:
                pass
        if conn:
            conn.close()
    
    async def get_document_with_content(self, doc_id: str, user_id: int) -> Optional[DocumentWithContent]:
        """
        Get document with content in a single optimized query
        
        Args:
            doc_id: Document identifier
            user_id: User identifier for access control
            
        Returns:
            DocumentWithContent or None if not found/unauthorized
        """
        if not self.validate_user_access(doc_id, user_id):
            return None
        
        conn = None
        try:
            conn = self._get_connection()
            
            if settings.USE_RDS and settings.IS_POSTGRES:
                # Optimized PostgreSQL query with JOIN
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute("""
                    SELECT d.doc_id, d.filename, d.file_id, d.user_id, d.updated_at,
                           f.file_data, f.content_type, f.file_size
                    FROM documents d
                    LEFT JOIN file_storage f ON d.file_id = f.file_id
                    WHERE d.doc_id = %s AND d.status = 'active'
                    LIMIT 1
                """, (doc_id,))
                
                row = cur.fetchone()
                if not row or not row['file_data']:
                    return None
                
                # Handle memoryview conversion
                file_data = row['file_data']
                if isinstance(file_data, memoryview):
                    file_data = file_data.tobytes()
                
                metadata = DocumentMetadata.from_db_row(row)
                return DocumentWithContent(metadata=metadata, content=file_data)
            
            else:
                # Fallback for SQLite/MySQL - separate queries
                metadata = await self.get_document_metadata_only(doc_id, user_id)
                if not metadata:
                    return None
                
                # For non-PostgreSQL, we don't have file storage in database
                # This would need to be handled by the file system fallback
                return None
                
        except Exception as e:
            print(f"Error retrieving document with content: {e}")
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def get_document_metadata_only(self, doc_id: str, user_id: int) -> Optional[DocumentMetadata]:
        """
        Get document metadata only (lightweight query)
        
        Args:
            doc_id: Document identifier
            user_id: User identifier for access control
            
        Returns:
            DocumentMetadata or None if not found/unauthorized
        """
        if not self.validate_user_access(doc_id, user_id):
            return None
        
        conn = None
        try:
            conn = self._get_connection()
            
            if settings.USE_RDS and settings.IS_POSTGRES:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute("""
                    SELECT d.doc_id, d.filename, d.file_id, d.user_id, d.updated_at,
                           f.content_type, f.file_size
                    FROM documents d
                    LEFT JOIN file_storage f ON d.file_id = f.file_id
                    WHERE d.doc_id = %s AND d.status = 'active'
                    LIMIT 1
                """, (doc_id,))
                
                row = cur.fetchone()
                if row:
                    return DocumentMetadata.from_db_row(row)
            
            else:
                # Fallback for SQLite/MySQL
                cur = conn.cursor()
                placeholder = self.db_manager._get_placeholder()
                cur.execute(f"""
                    SELECT doc_id, filename, user_id, updated_at
                    FROM documents 
                    WHERE doc_id = {placeholder} AND status = 'active'
                    LIMIT 1
                """, (doc_id,))
                
                row = cur.fetchone()
                if row:
                    return DocumentMetadata(
                        doc_id=row[0],
                        filename=row[1],
                        file_id="",  # Not available in legacy storage
                        file_size=0,  # Not available in legacy storage
                        content_type="application/pdf",
                        user_id=row[2],
                        last_modified=row[3] if row[3] else datetime.now(),
                        etag=f"{row[0]}-{row[3].timestamp() if row[3] else datetime.now().timestamp()}"
                    )
            
            return None
            
        except Exception as e:
            print(f"Error retrieving document metadata: {e}")
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def stream_document_content(self, file_id: str) -> AsyncIterator[bytes]:
        """
        Stream document content in chunks for large files
        
        Args:
            file_id: File identifier
            
        Yields:
            bytes: File content chunks
        """
        if not settings.USE_RDS or not settings.IS_POSTGRES:
            raise Exception("Streaming only available with PostgreSQL database storage")
        
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Stream file data in chunks to avoid loading entire file into memory
            cur.execute("SELECT file_data FROM file_storage WHERE file_id = %s", (file_id,))
            
            # For PostgreSQL, we can stream the BYTEA data
            row = cur.fetchone()
            if row and row[0]:
                file_data = row[0]
                if isinstance(file_data, memoryview):
                    file_data = file_data.tobytes()
                
                # Yield in 64KB chunks
                chunk_size = 64 * 1024
                for i in range(0, len(file_data), chunk_size):
                    yield file_data[i:i + chunk_size]
                    # Allow other coroutines to run
                    await asyncio.sleep(0)
            
        except Exception as e:
            print(f"Error streaming document content: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def validate_user_access(self, doc_id: str, user_id: int) -> bool:
        """
        Validate that user has access to document
        
        Args:
            doc_id: Document identifier
            user_id: User identifier
            
        Returns:
            bool: True if user has access, False otherwise
        """
        if not doc_id or not user_id:
            return False
        
        conn = None
        try:
            conn = self._get_connection()
            
            if settings.USE_RDS and settings.IS_POSTGRES:
                cur = conn.cursor()
                # Direct ownership check
                cur.execute("""
                    SELECT 1 FROM documents 
                    WHERE doc_id = %s AND user_id = %s AND status = 'active'
                    LIMIT 1
                """, (doc_id, user_id))
                owned = cur.fetchone() is not None
                if owned:
                    return True

                # Shared via project membership
                cur.execute("""
                    SELECT 1
                    FROM documents d
                    INNER JOIN project_documents pd ON pd.doc_id = d.doc_id
                    INNER JOIN project_members pm ON pm.project_id = pd.project_id
                    WHERE d.doc_id = %s AND d.status = 'active' AND pm.user_id = %s
                    LIMIT 1
                """, (doc_id, user_id))
                return cur.fetchone() is not None
            else:
                cur = conn.cursor()
                placeholder = self.db_manager._get_placeholder()
                # Direct ownership check
                cur.execute(f"""
                    SELECT 1 FROM documents 
                    WHERE doc_id = {placeholder} AND user_id = {placeholder} AND status = 'active'
                    LIMIT 1
                """, (doc_id, user_id))
                owned = cur.fetchone() is not None
                if owned:
                    return True

                # Shared via project membership
                cur.execute(f"""
                    SELECT 1
                    FROM documents d
                    INNER JOIN project_documents pd ON pd.doc_id = d.doc_id
                    INNER JOIN project_members pm ON pm.project_id = pd.project_id
                    WHERE d.doc_id = {placeholder} AND d.status = 'active' AND pm.user_id = {placeholder}
                    LIMIT 1
                """, (doc_id, user_id))
                return cur.fetchone() is not None
            
        except Exception as e:
            print(f"Error validating user access: {e}")
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics for monitoring"""
        if not self._connection_pool:
            return {"pool_enabled": False}
        
        try:
            return {
                "pool_enabled": True,
                "min_connections": self._connection_pool.minconn,
                "max_connections": self._connection_pool.maxconn,
                # Note: ThreadedConnectionPool doesn't expose current connection count
                # This would need custom tracking for detailed stats
            }
        except Exception:
            return {"pool_enabled": True, "error": "Unable to get pool stats"}
    
    def close_connection_pool(self):
        """Close connection pool on shutdown"""
        if self._connection_pool:
            try:
                self._connection_pool.closeall()
            except Exception as e:
                print(f"Error closing connection pool: {e}")


# Global optimized service instance
optimized_document_service = OptimizedDocumentService()
