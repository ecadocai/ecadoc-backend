"""
Document cache manager with LRU eviction for high-performance PDF downloads
"""
import asyncio
import sys
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
from modules.config.settings import settings
from modules.database.optimized_service import DocumentMetadata, DocumentWithContent


@dataclass
class CacheEntry:
    """Cache entry with access tracking"""
    data: Any
    access_time: datetime
    access_count: int = 0
    size_bytes: int = 0
    
    def update_access(self):
        """Update access time and count"""
        self.access_time = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    metadata_cache_size: int = 0
    file_cache_size: int = 0
    total_memory_usage: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_count: int = 0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def calculate_rates(self):
        """Calculate hit and miss rates"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
            self.miss_rate = self.cache_misses / self.total_requests
        else:
            self.hit_rate = 0.0
            self.miss_rate = 0.0


class DocumentCacheManager:
    """
    High-performance document cache with LRU eviction and memory management
    """
    
    def __init__(self, 
                 max_size: int = None, 
                 max_file_size_mb: int = None,
                 max_memory_mb: int = None):
        """
        Initialize cache manager
        
        Args:
            max_size: Maximum number of entries (default from settings)
            max_file_size_mb: Maximum file size to cache in MB (default from settings)
            max_memory_mb: Maximum total memory usage in MB (default from settings)
        """
        # Configuration
        self._max_size = max_size or getattr(settings, 'CACHE_MAX_SIZE', 1000)
        self._max_file_size = (max_file_size_mb or getattr(settings, 'CACHE_MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
        self._max_memory = (max_memory_mb or getattr(settings, 'CACHE_MAX_MEMORY_MB', 100)) * 1024 * 1024
        
        # Cache storage - using OrderedDict for LRU behavior
        self._metadata_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._file_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = CacheStats()
        self._start_time = datetime.now()
        
        # Memory tracking
        self._current_memory_usage = 0
        
        print(f"DocumentCacheManager initialized: max_size={self._max_size}, "
              f"max_file_size={self._max_file_size//1024//1024}MB, "
              f"max_memory={self._max_memory//1024//1024}MB")
    
    async def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata from cache
        
        Args:
            doc_id: Document identifier
            
        Returns:
            DocumentMetadata or None if not cached
        """
        with self._lock:
            self._stats.total_requests += 1
            
            cache_key = f"meta:{doc_id}"
            if cache_key in self._metadata_cache:
                entry = self._metadata_cache[cache_key]
                entry.update_access()
                
                # Move to end (most recently used)
                self._metadata_cache.move_to_end(cache_key)
                
                self._stats.cache_hits += 1
                self._stats.calculate_rates()
                
                return entry.data
            
            self._stats.cache_misses += 1
            self._stats.calculate_rates()
            return None
    
    async def get_document_content(self, doc_id: str) -> Optional[bytes]:
        """
        Get document content from cache
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document content bytes or None if not cached
        """
        with self._lock:
            self._stats.total_requests += 1
            
            cache_key = f"file:{doc_id}"
            if cache_key in self._file_cache:
                entry = self._file_cache[cache_key]
                entry.update_access()
                
                # Move to end (most recently used)
                self._file_cache.move_to_end(cache_key)
                
                self._stats.cache_hits += 1
                self._stats.calculate_rates()
                
                return entry.data
            
            self._stats.cache_misses += 1
            self._stats.calculate_rates()
            return None
    
    async def cache_document_metadata(self, doc_id: str, metadata: DocumentMetadata):
        """
        Cache document metadata
        
        Args:
            doc_id: Document identifier
            metadata: Document metadata to cache
        """
        with self._lock:
            cache_key = f"meta:{doc_id}"
            
            # Calculate size (rough estimate for metadata)
            size_bytes = sys.getsizeof(metadata) + len(metadata.filename) * 2 + len(metadata.etag) * 2
            
            # Check if we need to evict entries
            await self._ensure_capacity(size_bytes, is_metadata=True)
            
            # Create cache entry
            entry = CacheEntry(
                data=metadata,
                access_time=datetime.now(),
                access_count=1,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._metadata_cache[cache_key] = entry
            self._current_memory_usage += size_bytes
            
            # Update stats
            self._stats.metadata_cache_size = len(self._metadata_cache)
            self._stats.total_memory_usage = self._current_memory_usage
    
    async def cache_document_content(self, doc_id: str, content: bytes):
        """
        Cache document content if it meets size requirements
        
        Args:
            doc_id: Document identifier
            content: Document content bytes
        """
        # Check if file is too large to cache
        if len(content) > self._max_file_size:
            return
        
        with self._lock:
            cache_key = f"file:{doc_id}"
            size_bytes = len(content)
            
            # Check if we need to evict entries
            await self._ensure_capacity(size_bytes, is_metadata=False)
            
            # Create cache entry
            entry = CacheEntry(
                data=content,
                access_time=datetime.now(),
                access_count=1,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._file_cache[cache_key] = entry
            self._current_memory_usage += size_bytes
            
            # Update stats
            self._stats.file_cache_size = len(self._file_cache)
            self._stats.total_memory_usage = self._current_memory_usage
    
    async def cache_document(self, doc_id: str, metadata: DocumentMetadata, content: bytes = None):
        """
        Cache both metadata and content for a document
        
        Args:
            doc_id: Document identifier
            metadata: Document metadata
            content: Optional document content bytes
        """
        await self.cache_document_metadata(doc_id, metadata)
        
        if content is not None:
            await self.cache_document_content(doc_id, content)

    # ---------- Page Text Caching (lightweight helpers) ----------
    async def get_page_text(self, doc_id: str, page: int) -> Optional[str]:
        """Get cached page text for a document page if present."""
        with self._lock:
            self._stats.total_requests += 1
            key = f"text:{doc_id}:{page}"
            if key in self._metadata_cache:
                entry = self._metadata_cache[key]
                entry.update_access()
                self._metadata_cache.move_to_end(key)
                self._stats.cache_hits += 1
                self._stats.calculate_rates()
                return entry.data  # type: ignore[return-value]
            self._stats.cache_misses += 1
            self._stats.calculate_rates()
            return None

    async def cache_page_text(self, doc_id: str, page: int, text: str) -> None:
        """Cache text for a specific page of a document."""
        with self._lock:
            key = f"text:{doc_id}:{page}"
            size_bytes = len(text.encode("utf-8"))
            # Ensure capacity
            # Use metadata cache lane for small text entries
            # (best-effort; if eviction needed it will happen here)
            # Reuse _ensure_capacity indirectly via similar logic
            # Simplified: if metadata cache full, evict 1 LRU
            if len(self._metadata_cache) >= self._max_size:
                # Evict one LRU metadata entry
                try:
                    _, entry = self._metadata_cache.popitem(last=False)
                    self._current_memory_usage -= entry.size_bytes
                    self._stats.eviction_count += 1
                except Exception:
                    pass
            entry = CacheEntry(
                data=text,
                access_time=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
            )
            self._metadata_cache[key] = entry
            self._current_memory_usage += size_bytes
            self._stats.metadata_cache_size = len(self._metadata_cache)
            self._stats.total_memory_usage = self._current_memory_usage
    
    async def _ensure_capacity(self, required_bytes: int, is_metadata: bool):
        """
        Ensure cache has capacity for new entry by evicting LRU entries if needed
        
        Args:
            required_bytes: Bytes needed for new entry
            is_metadata: True if this is for metadata cache, False for file cache
        """
        # Check memory limit
        if self._current_memory_usage + required_bytes > self._max_memory:
            await self._evict_by_memory(required_bytes)
        
        # Check size limits
        target_cache = self._metadata_cache if is_metadata else self._file_cache
        if len(target_cache) >= self._max_size:
            await self._evict_lru_entries(1, is_metadata)
    
    async def _evict_by_memory(self, required_bytes: int):
        """
        Evict entries to free up memory
        
        Args:
            required_bytes: Bytes needed to be freed
        """
        bytes_to_free = (self._current_memory_usage + required_bytes) - self._max_memory
        bytes_freed = 0
        
        # Evict from file cache first (larger entries)
        while bytes_freed < bytes_to_free and self._file_cache:
            cache_key, entry = self._file_cache.popitem(last=False)  # Remove oldest
            bytes_freed += entry.size_bytes
            self._current_memory_usage -= entry.size_bytes
            self._stats.eviction_count += 1
        
        # If still need more space, evict from metadata cache
        while bytes_freed < bytes_to_free and self._metadata_cache:
            cache_key, entry = self._metadata_cache.popitem(last=False)  # Remove oldest
            bytes_freed += entry.size_bytes
            self._current_memory_usage -= entry.size_bytes
            self._stats.eviction_count += 1
        
        # Update stats
        self._stats.metadata_cache_size = len(self._metadata_cache)
        self._stats.file_cache_size = len(self._file_cache)
        self._stats.total_memory_usage = self._current_memory_usage
    
    async def _evict_lru_entries(self, count: int, is_metadata: bool):
        """
        Evict least recently used entries
        
        Args:
            count: Number of entries to evict
            is_metadata: True to evict from metadata cache, False from file cache
        """
        target_cache = self._metadata_cache if is_metadata else self._file_cache
        
        for _ in range(min(count, len(target_cache))):
            if target_cache:
                cache_key, entry = target_cache.popitem(last=False)  # Remove oldest
                self._current_memory_usage -= entry.size_bytes
                self._stats.eviction_count += 1
        
        # Update stats
        if is_metadata:
            self._stats.metadata_cache_size = len(self._metadata_cache)
        else:
            self._stats.file_cache_size = len(self._file_cache)
        self._stats.total_memory_usage = self._current_memory_usage
    
    def evict_document(self, doc_id: str):
        """
        Manually evict a specific document from cache
        
        Args:
            doc_id: Document identifier to evict
        """
        with self._lock:
            meta_key = f"meta:{doc_id}"
            file_key = f"file:{doc_id}"
            
            # Remove metadata
            if meta_key in self._metadata_cache:
                entry = self._metadata_cache.pop(meta_key)
                self._current_memory_usage -= entry.size_bytes
                self._stats.eviction_count += 1
            
            # Remove file content
            if file_key in self._file_cache:
                entry = self._file_cache.pop(file_key)
                self._current_memory_usage -= entry.size_bytes
                self._stats.eviction_count += 1
            
            # Update stats
            self._stats.metadata_cache_size = len(self._metadata_cache)
            self._stats.file_cache_size = len(self._file_cache)
            self._stats.total_memory_usage = self._current_memory_usage
    
    def clear_cache(self):
        """Clear all cached entries"""
        with self._lock:
            old_meta_size = len(self._metadata_cache)
            old_file_size = len(self._file_cache)
            
            self._metadata_cache.clear()
            self._file_cache.clear()
            self._current_memory_usage = 0
            self._stats.eviction_count += old_meta_size + old_file_size
            
            # Update stats
            self._stats.metadata_cache_size = 0
            self._stats.file_cache_size = 0
            self._stats.total_memory_usage = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            uptime = datetime.now() - self._start_time
            
            return {
                "metadata_cache_size": self._stats.metadata_cache_size,
                "file_cache_size": self._stats.file_cache_size,
                "total_entries": self._stats.metadata_cache_size + self._stats.file_cache_size,
                "max_size": self._max_size,
                "memory_usage_bytes": self._stats.total_memory_usage,
                "memory_usage_mb": self._stats.total_memory_usage / (1024 * 1024),
                "max_memory_mb": self._max_memory / (1024 * 1024),
                "memory_usage_percent": (self._stats.total_memory_usage / self._max_memory) * 100,
                "hit_rate": self._stats.hit_rate,
                "miss_rate": self._stats.miss_rate,
                "total_requests": self._stats.total_requests,
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "eviction_count": self._stats.eviction_count,
                "uptime_seconds": uptime.total_seconds(),
                "max_file_size_mb": self._max_file_size / (1024 * 1024)
            }
    
    def get_cache_entries_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about cached entries (for debugging)
        
        Returns:
            List of cache entry information
        """
        with self._lock:
            entries = []
            
            # Metadata entries
            for key, entry in self._metadata_cache.items():
                doc_id = key.replace("meta:", "")
                entries.append({
                    "doc_id": doc_id,
                    "type": "metadata",
                    "size_bytes": entry.size_bytes,
                    "access_count": entry.access_count,
                    "access_time": entry.access_time.isoformat(),
                    "filename": entry.data.filename if hasattr(entry.data, 'filename') else None
                })
            
            # File entries
            for key, entry in self._file_cache.items():
                doc_id = key.replace("file:", "")
                entries.append({
                    "doc_id": doc_id,
                    "type": "file_content",
                    "size_bytes": entry.size_bytes,
                    "size_mb": entry.size_bytes / (1024 * 1024),
                    "access_count": entry.access_count,
                    "access_time": entry.access_time.isoformat()
                })
            
            # Sort by access time (most recent first)
            entries.sort(key=lambda x: x["access_time"], reverse=True)
            
            return entries
    
    async def warm_cache(self, doc_ids: List[str], optimized_service):
        """
        Warm cache with frequently accessed documents
        
        Args:
            doc_ids: List of document IDs to pre-load
            optimized_service: OptimizedDocumentService instance
        """
        for doc_id in doc_ids:
            try:
                # This would need user_id - in practice, this might be called
                # with a list of (doc_id, user_id) tuples
                # For now, we'll skip the actual warming but keep the structure
                pass
            except Exception as e:
                print(f"Failed to warm cache for doc_id {doc_id}: {e}")


# Global cache manager instance
document_cache_manager = DocumentCacheManager()
