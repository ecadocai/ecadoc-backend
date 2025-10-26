"""
PDF processing and document management for the Floor Plan Agent API
"""
import os
import uuid
from datetime import datetime
import io
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from modules.config.settings import settings
from modules.database.models import db_manager

class PDFProcessor:
    """PDF processing and indexing service"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.use_database_storage = settings.USE_RDS and settings.IS_POSTGRES

    def _get_user_storage_limit_mb(self, user_id: int) -> float:
        """Calculate storage limit in megabytes for a user."""
        subscription = db_manager.get_user_subscription(user_id)
        if subscription:
            plan = db_manager.get_subscription_plan_by_id(subscription.plan_id)
            if plan:
                return plan.storage_gb * 1024

        user = db_manager.get_user_by_id(user_id)
        created_at = getattr(user, "created_at", None) if user else None
        if created_at:
            try:
                timestamp = created_at
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                if timestamp and (datetime.now() - timestamp).days <= 30:
                    return 15 * 1024
            except Exception:
                pass

        return 100

    def _calculate_document_size_mb(self, document: Any) -> float:
        """Estimate the size of a stored document in megabytes."""
        if not document:
            return 0.0

        size_bytes = 0

        if self.use_database_storage and getattr(document, "file_id", None):
            try:
                file_record = db_manager.get_file(document.file_id)
                if file_record and getattr(file_record, "file_size", None):
                    size_bytes = file_record.file_size
            except Exception:
                size_bytes = 0
        else:
            try:
                pdf_path = os.path.join(settings.DOCS_DIR, f"{document.doc_id}.pdf")
                if os.path.exists(pdf_path):
                    size_bytes = os.path.getsize(pdf_path)
            except Exception:
                size_bytes = 0

        return size_bytes / (1024 * 1024) if size_bytes else 0.0

    def _decrement_user_storage(self, user_id: int, file_size_mb: float) -> None:
        """Reduce a user's recorded storage usage after deletions."""
        if file_size_mb <= 0:
            return

        try:
            storage = db_manager.get_user_storage(user_id)
            if not storage:
                return

            new_usage = max(0.0, storage.used_storage_mb - file_size_mb)
            db_manager.update_user_storage(user_id, new_usage)
        except Exception:
            # Avoid raising during cleanup routines
            pass

    def pdf_to_documents(self, pdf_source, doc_id: str) -> List[Document]:
        """Convert PDF to document chunks for indexing
        
        Args:
            pdf_source: Either a file path (str) or bytes data
            doc_id: Document identifier
        """
        if isinstance(pdf_source, bytes):
            # Read from bytes (database storage)
            reader = PdfReader(io.BytesIO(pdf_source))
        else:
            # Read from file path (legacy file storage)
            reader = PdfReader(pdf_source)
        
        docs: List[Document] = []
        
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                # Keep empty pages indexable for page grounding
                text = f"[No text extracted: page {i}]"
            
            for chunk in self.text_splitter.split_text(text):
                docs.append(Document(
                    page_content=chunk, 
                    metadata={"doc_id": doc_id, "page": i}
                ))
        
        return docs
    
    def pdf_bytes_to_documents(self, pdf_data: bytes, doc_id: str) -> List[Document]:
        """Convert PDF bytes to document chunks for indexing"""
        return self.pdf_to_documents(pdf_data, doc_id)
    
    def index_pdf(self, doc_id: str, pdf_source) -> int:
        """Index PDF into vector store
        
        Args:
            doc_id: Document identifier
            pdf_source: Either file path (str) or bytes data
        """
        docs = self.pdf_to_documents(pdf_source, doc_id)
        if not docs:
            raise ValueError("No text extracted from PDF")
        
        if self.use_database_storage:
            # Store vectors in database using pgvector
            return self.index_pdf_to_database(doc_id, docs)
        else:
            # Store vectors in local FAISS files (legacy)
            vs = FAISS.from_documents(docs, self.embeddings)
            vs.save_local(os.path.join(settings.VECTORS_DIR, doc_id))
            return len(docs)
    
    def index_pdf_to_database(self, doc_id: str, docs: List[Document]) -> int:
        """Index PDF documents to database using pgvector"""
        if not self.use_database_storage:
            raise Exception("Database storage not available")
        
        # Generate embeddings for all chunks
        texts = [doc.page_content for doc in docs]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare chunks for database storage
        chunks = []
        for doc, embedding in zip(docs, embeddings):
            chunks.append({
                'text': doc.page_content,
                'page': doc.metadata.get('page', 1),
                'embedding': embedding
            })
        
        # Store in database
        stored_count = db_manager.store_vector_chunks(doc_id, chunks)
        return stored_count
    
    def load_vectorstore(self, doc_id: str) -> FAISS:
        """Load vector store for a document (legacy file storage)"""
        if self.use_database_storage:
            raise Exception("Use query_document_vectors for database storage")
        
        path = os.path.join(settings.VECTORS_DIR, doc_id)
        if not os.path.exists(path):
            raise FileNotFoundError("Vectorstore for doc not found.")
        
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
    
    def get_document_content(self, doc_id: str) -> bytes:
        """Get document content from database"""
        if not self.use_database_storage:
            raise Exception("Document content retrieval only available with database storage")
        
        document = db_manager.get_document_by_doc_id(doc_id)
        if not document:
            raise FileNotFoundError("Document not found")
        
        file_storage = db_manager.get_file(document.file_id)
        if not file_storage:
            raise FileNotFoundError("Document file not found")
        
        return file_storage.file_data
    
    def query_document_vectors(self, doc_id: str, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query document using database vector storage"""
        if not self.use_database_storage:
            raise Exception("Vector querying only available with database storage")
        
        # Generate embedding for the question
        query_embedding = self.embeddings.embed_query(question)
        
        # Perform similarity search
        results = db_manager.similarity_search(doc_id, query_embedding, k)
        
        return results
    
    def delete_document_completely(self, doc_id: str) -> bool:
        """Delete document and all associated data"""
        try:
            document = db_manager.get_document_by_doc_id(doc_id)
            if not document:
                return False

            file_size_mb = self._calculate_document_size_mb(document)

            if self.use_database_storage:
                # Delete from database storage
                success = True

                # Delete vector chunks
                try:
                    db_manager.delete_vector_chunks(doc_id)
                except:
                    success = False
                
                # Delete file
                if document.file_id:
                    try:
                        db_manager.delete_file(document.file_id)
                    except:
                        success = False
                
                # Delete document record
                try:
                    db_manager.delete_document(doc_id)
                except:
                    success = False

                if success:
                    self._decrement_user_storage(document.user_id, file_size_mb)
                    db_manager.log_user_activity(
                        document.user_id,
                        "document_deleted",
                        {
                            "doc_id": doc_id,
                            "filename": document.filename,
                            "file_size_mb": round(file_size_mb, 2),
                        },
                    )

                return success
            else:
                # Delete from file storage (legacy)
                return self.delete_document_files(doc_id)

        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
    
    def store_generated_output(self, output_data: bytes, filename: str, source_doc_id: str, 
                              user_id: int, metadata: Dict[str, Any] = None) -> str:
        """Store generated output file and return output_id"""
        if not self.use_database_storage:
            raise Exception("Generated output storage only available with database storage")
        
        output_id = str(uuid.uuid4())
        
        success = db_manager.store_generated_output(
            output_id=output_id,
            filename=filename,
            content_type="application/pdf",
            file_data=output_data,
            source_doc_id=source_doc_id,
            user_id=user_id,
            metadata=metadata
        )
        
        if success:
            return output_id
        else:
            raise Exception("Failed to store generated output")
    
    def get_generated_output(self, output_id: str) -> Dict[str, Any]:
        """Get generated output by ID"""
        if not self.use_database_storage:
            raise Exception("Generated output retrieval only available with database storage")
        
        output = db_manager.get_generated_output(output_id)
        if not output:
            raise FileNotFoundError("Generated output not found")
        
        return output.to_dict(include_data=True)
    
    def list_user_generated_outputs(self, user_id: int) -> List[Dict[str, Any]]:
        """List all generated outputs for a user"""
        if not self.use_database_storage:
            raise Exception("Generated output listing only available with database storage")
        
        outputs = db_manager.list_generated_outputs(user_id)
        return [output.to_dict(include_data=False) for output in outputs]
    
    def delete_generated_output(self, output_id: str) -> bool:
        """Delete generated output by ID"""
        if not self.use_database_storage:
            raise Exception("Generated output deletion only available with database storage")
        
        return db_manager.delete_generated_output(output_id)
    
    def save_output_to_database_or_file(self, output_data: bytes, filename: str, 
                                       source_doc_id: str, user_id: int, 
                                       metadata: Dict[str, Any] = None) -> str:
        """Save output to database if available, otherwise to file system"""
        if self.use_database_storage:
            # Store in database
            return self.store_generated_output(output_data, filename, source_doc_id, user_id, metadata)
        else:
            # Store in file system (legacy)
            if settings.OUTPUT_DIR is None:
                import tempfile
                output_path = os.path.join(tempfile.gettempdir(), filename)
            else:
                output_path = os.path.join(settings.OUTPUT_DIR, filename)
                os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(output_data)
            
            return output_path
    
    def upload_and_index_pdf(self, file_content: bytes, filename: str, user_id: int) -> dict:
        """Upload and index a PDF file"""
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Please upload a PDF file")
        
        doc_id = uuid.uuid4().hex

        file_size_mb = len(file_content) / (1024 * 1024)
        user_storage = db_manager.get_user_storage(user_id)
        if not user_storage:
            db_manager.create_user_storage(user_id)
            user_storage = db_manager.get_user_storage(user_id)

        storage_limit_mb = self._get_user_storage_limit_mb(user_id)
        if user_storage and user_storage.used_storage_mb + file_size_mb > storage_limit_mb:
            available = storage_limit_mb - user_storage.used_storage_mb
            raise ValueError(
                f"Storage limit exceeded. Available: {max(0, available):.2f}MB, Required: {file_size_mb:.2f}MB"
            )
        
        if self.use_database_storage:
            # Store file in database
            file_id = str(uuid.uuid4())
            
            # Store file in database
            db_manager.store_file(
                file_id=file_id,
                filename=filename,
                content_type="application/pdf",
                file_data=file_content,
                user_id=user_id
            )
        else:
            # Store file locally (legacy)
            pdf_path = os.path.join(settings.DOCS_DIR, f"{doc_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(file_content)
            file_id = None
        
        # Get page count first
        if self.use_database_storage:
            reader = PdfReader(io.BytesIO(file_content))
        else:
            reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        # Create document record FIRST (required for foreign key constraint)
        try:
            if self.use_database_storage:
                db_manager.create_document(
                    doc_id=doc_id,
                    filename=filename,
                    file_id=file_id,
                    pages=num_pages,
                    chunks_indexed=0,  # Will be updated after indexing
                    user_id=user_id
                )
            else:
                # Legacy file storage
                vector_path = os.path.join(settings.VECTORS_DIR, doc_id)
                db_manager.create_document(
                    doc_id=doc_id,
                    filename=filename,
                    file_id="",
                    pages=num_pages,
                    chunks_indexed=0,  # Will be updated after indexing
                    user_id=user_id,
                    pdf_path=pdf_path,
                    vector_path=vector_path
                )
        except Exception as e:
            # Clean up file on document creation failure
            if self.use_database_storage:
                db_manager.delete_file(file_id)
            else:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            raise ValueError(f"Document creation failed: {e}")
        
        # Now index into vector store (document record exists for foreign key)
        try:
            if self.use_database_storage:
                n_chunks = self.index_pdf(doc_id, file_content)
            else:
                n_chunks = self.index_pdf(doc_id, pdf_path)

            # Update document with actual chunk count
            db_manager.update_document_chunks_indexed(doc_id, n_chunks)

            latest_storage = db_manager.get_user_storage(user_id)
            base_usage = latest_storage.used_storage_mb if latest_storage else 0
            db_manager.update_user_storage(user_id, base_usage + file_size_mb)
            db_manager.log_user_activity(
                user_id,
                "document_uploaded",
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_size_mb": round(file_size_mb, 2),
                }
            )
            
        except Exception as e:
            # Clean up on indexing failure
            if self.use_database_storage:
                db_manager.delete_file(file_id)
                db_manager.delete_document(doc_id)
            else:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                vector_path = os.path.join(settings.VECTORS_DIR, doc_id)
                if os.path.exists(vector_path):
                    import shutil
                    shutil.rmtree(vector_path)
                db_manager.delete_document(doc_id)
            raise ValueError(f"Indexing failed: {e}")
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "pages": num_pages,
            "chunks_indexed": n_chunks
        }
    
    def upload_and_index_multiple_pdfs(self, file_contents: List[bytes], filenames: List[str], user_id: int) -> List[dict]:
        """Upload and index multiple PDF files"""
        results = []
        errors = []
        
        for i, (file_content, filename) in enumerate(zip(file_contents, filenames)):
            try:
                result = self.upload_and_index_pdf(file_content, filename, user_id)
                results.append(result)
            except Exception as e:
                error_info = {
                    "filename": filename,
                    "error": str(e),
                    "index": i
                }
                errors.append(error_info)
        
        return {
            "successful_uploads": results,
            "failed_uploads": errors,
            "total_files": len(file_contents),
            "successful_count": len(results),
            "failed_count": len(errors)
        }
    
    def get_document_info(self, doc_id: str) -> dict:
        """Get document information"""
        document = db_manager.get_document_by_doc_id(doc_id)
        if not document:
            raise FileNotFoundError("Document not found")
        
        if self.use_database_storage:
            # For database storage, include metadata from the file storage table
            file_record = None
            try:
                if document.file_id:
                    file_record = db_manager.get_file(document.file_id)
            except Exception:
                file_record = None

            file_size_bytes = getattr(file_record, "file_size", None) if file_record else None

            return {
                "doc_id": doc_id,
                "filename": document.filename,
                "file_id": document.file_id,
                "pages": document.pages,
                "chunks_indexed": document.chunks_indexed,
                "status": document.status,
                "storage_type": "database",
                "file_exists": bool(file_record),
                "file_size_bytes": file_size_bytes,
                "user_id": document.user_id,
            }
        else:
            # Legacy file storage - derive filesystem paths for compatibility
            pdf_path = os.path.join(settings.DOCS_DIR, f"{doc_id}.pdf")
            vector_path = os.path.join(settings.VECTORS_DIR, doc_id)
            file_size_bytes: Optional[int] = None

            try:
                if os.path.exists(pdf_path):
                    file_size_bytes = os.path.getsize(pdf_path)
            except Exception:
                file_size_bytes = None

            return {
                "doc_id": doc_id,
                "filename": document.filename,
                "pages": document.pages,
                "chunks_indexed": document.chunks_indexed,
                "status": document.status,
                "storage_type": "filesystem",
                "pdf_path": pdf_path,
                "vector_path": vector_path,
                "file_size_bytes": file_size_bytes,
                "user_id": document.user_id,
            }
    
    def list_documents(self, user_id: int = None) -> dict:
        """List all documents in the database"""
        if user_id:
            documents = db_manager.get_user_documents(user_id)
        else:
            documents = db_manager.get_all_documents()
        
        # Convert to dictionary format
        result = {}
        for doc in documents:
            doc_info = {
                "filename": doc.filename,
                "pages": doc.pages,
                "chunks_indexed": doc.chunks_indexed,
                "status": doc.status,
                "user_id": doc.user_id,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
            }
            
            if self.use_database_storage:
                # For database storage, add file_id and storage type
                doc_info["file_id"] = doc.file_id
                doc_info["storage_type"] = "database"
            else:
                # For legacy file storage, would need to check file existence
                doc_info["storage_type"] = "filesystem"
            
            result[doc.doc_id] = doc_info
        
        return result
    
    def query_document(self, doc_id: str, question: str, k: int = 5) -> dict:
        """Query document using RAG"""
        # Check if document exists in database
        document = db_manager.get_document_by_doc_id(doc_id)
        if not document:
            raise FileNotFoundError("Document not found")
        
        try:
            if self.use_database_storage:
                # Use database vector storage
                results = self.query_document_vectors(doc_id, question, k)
                
                return {
                    "doc_id": doc_id,
                    "question": question,
                    "matches": [
                        {
                            "page": result.get("page"),
                            "text": result.get("text")
                        } for result in results
                    ]
                }
            else:
                # Use legacy file storage - need to get paths from document info
                doc_info = self.get_document_info(doc_id)
                pdf_path = doc_info.get("pdf_path")
                vector_path = doc_info.get("vector_path")
                
                if not pdf_path or not os.path.exists(pdf_path):
                    db_manager.update_document_status(doc_id, "missing_pdf")
                    raise FileNotFoundError("PDF file not found")
                
                # Check if vector store exists
                if not vector_path or not os.path.exists(vector_path):
                    db_manager.update_document_status(doc_id, "missing_vectors")
                    raise FileNotFoundError("Vector store not found")
                
                vs = self.load_vectorstore(doc_id)
                docs = vs.similarity_search(question, k=int(k))
                
                return {
                    "doc_id": doc_id,
                    "question": question,
                    "matches": [
                        {
                            "page": d.metadata.get("page"),
                            "text": d.page_content
                        } for d in docs
                    ]
                }
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")
    
    def query_document_with_citations(self, doc_id: str, question: str, k: int = 5) -> dict:
        """Query document using RAG with detailed citation information"""
        # Check if document exists in database
        document = db_manager.get_document_by_doc_id(doc_id)
        if not document:
            raise FileNotFoundError("Document not found")
        
        # Get document paths from document info (handles both storage types)
        doc_info = self.get_document_info(doc_id)
        
        if self.use_database_storage:
            # For database storage, we don't need to check file paths
            pass
        else:
            # For legacy file storage, check if files exist
            pdf_path = doc_info.get("pdf_path")
            vector_path = doc_info.get("vector_path")
            
            if not pdf_path or not os.path.exists(pdf_path):
                db_manager.update_document_status(doc_id, "missing_pdf")
                raise FileNotFoundError("PDF file not found")
            
            if not vector_path or not os.path.exists(vector_path):
                db_manager.update_document_status(doc_id, "missing_vectors")
                raise FileNotFoundError("Vector store not found")
        
        try:
            vs = self.load_vectorstore(doc_id)
            docs = vs.similarity_search_with_score(question, k=int(k))
            
            # Process documents and organize by page
            citations_by_page = {}
            all_citations = []
            
            for i, (doc, score) in enumerate(docs):
                page_num = doc.metadata.get("page", 1)
                citation = {
                    "id": i + 1,
                    "page": page_num,
                    "text": doc.page_content,
                    "relevance_score": float(score),
                    "doc_id": doc_id
                }
                
                all_citations.append(citation)
                
                if page_num not in citations_by_page:
                    citations_by_page[page_num] = []
                citations_by_page[page_num].append(citation)
            
            # Find the most referenced page (page with most chunks)
            most_referenced_page = None
            max_citations = 0
            
            if citations_by_page:
                for page_num, page_citations in citations_by_page.items():
                    if len(page_citations) > max_citations:
                        max_citations = len(page_citations)
                        most_referenced_page = page_num
            
            return {
                "doc_id": doc_id,
                "question": question,
                "citations": all_citations,
                "citations_by_page": citations_by_page,
                "most_referenced_page": most_referenced_page,
                "total_citations": len(all_citations),
                "pages_referenced": list(citations_by_page.keys()) if citations_by_page else []
            }
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")
    
    def get_page_citations_summary(self, doc_id: str, question: str, k: int = 8) -> dict:
        """Get a summary of which pages are most relevant to a query"""
        try:
            result = self.query_document_with_citations(doc_id, question, k)
            
            # Count citations per page and calculate relevance scores
            page_summary = {}
            for page_num, citations in result["citations_by_page"].items():
                total_relevance = sum(c["relevance_score"] for c in citations)
                avg_relevance = total_relevance / len(citations) if citations else 0
                
                page_summary[page_num] = {
                    "page": page_num,
                    "citation_count": len(citations),
                    "total_relevance_score": total_relevance,
                    "avg_relevance_score": avg_relevance,
                    "citations": citations
                }
            
            # Sort pages by citation count, then by average relevance
            sorted_pages = sorted(
                page_summary.values(),
                key=lambda x: (x["citation_count"], x["avg_relevance_score"]),
                reverse=True
            )
            
            return {
                "doc_id": doc_id,
                "question": question,
                "most_relevant_page": sorted_pages[0]["page"] if sorted_pages else None,
                "page_rankings": sorted_pages,
                "total_pages_referenced": len(sorted_pages)
            }
        except Exception as e:
            raise ValueError(f"Citation summary failed: {str(e)}")
    
    def delete_document_files(self, doc_id: str) -> bool:
        """Delete both PDF and vector files for a document"""
        success = True

        # Get document info from database
        document = db_manager.get_document_by_doc_id(doc_id)
        if not document:
            return False

        file_size_mb = self._calculate_document_size_mb(document)

        # Handle file deletion based on storage type
        if self.use_database_storage:
            # For database storage, files are stored in database, no filesystem cleanup needed
            print(f"Document {doc_id} uses database storage, no filesystem cleanup needed")
        else:
            # For legacy file storage, delete files from filesystem
            doc_info = self.get_document_info(doc_id)
            pdf_path = doc_info.get("pdf_path")
            vector_path = doc_info.get("vector_path")
            
            # Delete PDF file
            try:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print(f"Deleted PDF file: {pdf_path}")
            except Exception as e:
                print(f"Error deleting PDF file {pdf_path}: {e}")
                success = False
            
            # Delete vector store directory
            try:
                if vector_path and os.path.exists(vector_path):
                    import shutil
                    shutil.rmtree(vector_path)
                    print(f"Deleted vector store: {vector_path}")
            except Exception as e:
                print(f"Error deleting vector store {vector_path}: {e}")
                success = False
        
        # Delete from database
        try:
            db_manager.delete_document(doc_id)
            print(f"Deleted document from database: {doc_id}")
        except Exception as e:
            print(f"Error deleting document from database {doc_id}: {e}")
            success = False

        if success:
            self._decrement_user_storage(document.user_id, file_size_mb)
            db_manager.log_user_activity(
                document.user_id,
                "document_deleted",
                {
                    "doc_id": doc_id,
                    "filename": document.filename,
                    "file_size_mb": round(file_size_mb, 2),
                },
            )

        return success

# Global PDF processor instance
pdf_processor = PDFProcessor()