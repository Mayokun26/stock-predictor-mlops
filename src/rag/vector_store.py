#!/usr/bin/env python3
"""
Vector Store for RAG (Retrieval-Augmented Generation)
Implements document embeddings and semantic search for market research
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from sklearn.metrics.pairwise import cosine_similarity
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class DocumentStore:
    """Document storage and retrieval for RAG system"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.model_name = "all-MiniLM-L6-v2"  # Fast, efficient embeddings model
        self.model = None
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2
        self.chunk_size = 512  # Characters per chunk
        self.overlap_size = 50   # Overlap between chunks
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Embedding model {self.model_name} loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.model = None
    
    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks"""
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.overlap_size):
            chunk_text = text[i:i + self.chunk_size]
            
            # Don't create tiny chunks at the end
            if len(chunk_text) < 50:
                break
                
            chunk = {
                "text": chunk_text,
                "chunk_id": i // (self.chunk_size - self.overlap_size),
                "start_pos": i,
                "end_pos": i + len(chunk_text),
                "metadata": metadata or {}
            }
            chunks.append(chunk)
            
        return chunks
    
    async def add_document(self, 
                          document_id: str,
                          text: str,
                          metadata: Dict[str, Any] = None,
                          source: str = "unknown") -> int:
        """Add document to vector store"""
        if not self.model:
            await self.initialize()
            if not self.model:
                raise Exception("Embedding model not available")
        
        # Create document metadata
        doc_metadata = {
            "document_id": document_id,
            "source": source,
            "added_at": datetime.utcnow().isoformat(),
            "text_length": len(text),
            **(metadata or {})
        }
        
        # Chunk the document
        chunks = self.chunk_document(text, doc_metadata)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(chunk_texts)
        
        # Store in Redis
        chunks_added = 0
        for chunk, embedding in zip(chunks, embeddings):
            chunk_key = f"doc_chunk:{document_id}:{chunk['chunk_id']}"
            
            chunk_data = {
                **chunk,
                "embedding": embedding.tolist(),  # Store as list for JSON serialization
                "embedding_dim": len(embedding),
                "model_used": self.model_name
            }
            
            if self.redis_client:
                await self.redis_client.setex(
                    chunk_key,
                    86400 * 7,  # 7 days TTL
                    json.dumps(chunk_data)
                )
            
            chunks_added += 1
        
        # Store document metadata
        doc_key = f"doc_meta:{document_id}"
        if self.redis_client:
            await self.redis_client.setex(
                doc_key,
                86400 * 7,  # 7 days TTL
                json.dumps({
                    **doc_metadata,
                    "total_chunks": chunks_added
                })
            )
        
        logger.info(f"✅ Added document {document_id}: {chunks_added} chunks")
        return chunks_added
    
    async def search_similar(self, 
                           query: str, 
                           top_k: int = 5,
                           similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar document chunks"""
        if not self.model:
            await self.initialize()
            if not self.model:
                return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Get all document chunks from Redis
        chunks = []
        if self.redis_client:
            async for key in self.redis_client.scan_iter(match="doc_chunk:*"):
                chunk_data = await self.redis_client.get(key)
                if chunk_data:
                    chunk = json.loads(chunk_data)
                    chunk["redis_key"] = key.decode()
                    chunks.append(chunk)
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            chunk_embedding = np.array(chunk["embedding"])
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            
            if similarity >= similarity_threshold:
                similarities.append({
                    "chunk": chunk,
                    "similarity": float(similarity),
                    "relevance_score": float(similarity)
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        chunks = []
        if not self.redis_client:
            return chunks
            
        async for key in self.redis_client.scan_iter(match=f"doc_chunk:{document_id}:*"):
            chunk_data = await self.redis_client.get(key)
            if chunk_data:
                chunk = json.loads(chunk_data)
                chunks.append(chunk)
        
        # Sort by chunk_id
        chunks.sort(key=lambda x: x.get("chunk_id", 0))
        return chunks
    
    async def delete_document(self, document_id: str) -> int:
        """Delete document and all its chunks"""
        if not self.redis_client:
            return 0
            
        deleted = 0
        
        # Delete all chunks
        async for key in self.redis_client.scan_iter(match=f"doc_chunk:{document_id}:*"):
            await self.redis_client.delete(key)
            deleted += 1
        
        # Delete metadata
        meta_key = f"doc_meta:{document_id}"
        if await self.redis_client.exists(meta_key):
            await self.redis_client.delete(meta_key)
            deleted += 1
        
        logger.info(f"🗑️ Deleted document {document_id}: {deleted} items")
        return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.redis_client:
            return {"status": "redis_unavailable"}
        
        stats = {
            "total_chunks": 0,
            "total_documents": 0,
            "embedding_model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size
        }
        
        # Count chunks
        async for key in self.redis_client.scan_iter(match="doc_chunk:*"):
            stats["total_chunks"] += 1
        
        # Count documents
        async for key in self.redis_client.scan_iter(match="doc_meta:*"):
            stats["total_documents"] += 1
        
        return stats

class MarketDataCollector:
    """Collect market-related documents for RAG"""
    
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
    async def collect_news_articles(self, 
                                  symbols: List[str], 
                                  days_back: int = 7) -> int:
        """Collect recent news articles about symbols"""
        collected = 0
        
        for symbol in symbols:
            try:
                articles = await self._fetch_news_for_symbol(symbol, days_back)
                
                for article in articles:
                    document_id = f"news_{symbol}_{article['publishedAt']}"
                    document_id = hashlib.md5(document_id.encode()).hexdigest()
                    
                    text = f"{article['title']}\n\n{article.get('description', '')}\n\n{article.get('content', '')}"
                    
                    metadata = {
                        "symbol": symbol,
                        "source": article.get('source', {}).get('name', 'unknown'),
                        "published_at": article['publishedAt'],
                        "url": article.get('url', ''),
                        "data_type": "news"
                    }
                    
                    await self.document_store.add_document(
                        document_id=document_id,
                        text=text,
                        metadata=metadata,
                        source="news_api"
                    )
                    collected += 1
                    
            except Exception as e:
                logger.error(f"Error collecting news for {symbol}: {e}")
        
        return collected
    
    async def _fetch_news_for_symbol(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        """Fetch news articles for a specific symbol"""
        if not self.news_api_key:
            # Return mock data for testing
            return [{
                "title": f"Sample news about {symbol}",
                "description": f"This is sample market news about {symbol} stock performance.",
                "content": f"Detailed analysis of {symbol} market trends and future outlook.",
                "publishedAt": datetime.utcnow().isoformat(),
                "source": {"name": "MockNews"},
                "url": f"https://example.com/news/{symbol}"
            }]
        
        # Real NewsAPI implementation would go here
        return []
    
    async def add_financial_reports(self, 
                                   symbol: str, 
                                   report_text: str,
                                   report_type: str = "earnings") -> str:
        """Add financial report to vector store"""
        document_id = f"report_{symbol}_{report_type}_{datetime.now().strftime('%Y%m%d')}"
        document_id = hashlib.md5(document_id.encode()).hexdigest()
        
        metadata = {
            "symbol": symbol,
            "report_type": report_type,
            "report_date": datetime.utcnow().isoformat(),
            "data_type": "financial_report"
        }
        
        await self.document_store.add_document(
            document_id=document_id,
            text=report_text,
            metadata=metadata,
            source="financial_reports"
        )
        
        return document_id

class RAGService:
    """Complete RAG service for enhanced market research"""
    
    def __init__(self, redis_client=None):
        self.document_store = DocumentStore(redis_client)
        self.market_collector = MarketDataCollector(self.document_store)
        
    async def initialize(self):
        """Initialize the RAG service"""
        await self.document_store.initialize()
        logger.info("✅ RAG service initialized")
    
    async def enhance_market_query(self, 
                                 query: str, 
                                 symbols: List[str] = None,
                                 max_contexts: int = 3) -> Dict[str, Any]:
        """Enhance market query with relevant context from vector store"""
        
        # Search for relevant documents
        similar_docs = await self.document_store.search_similar(
            query=query,
            top_k=max_contexts,
            similarity_threshold=0.3
        )
        
        contexts = []
        sources = set()
        
        for doc in similar_docs:
            chunk = doc["chunk"]
            contexts.append({
                "text": chunk["text"][:500],  # Limit context length
                "source": chunk["metadata"].get("source", "unknown"),
                "symbol": chunk["metadata"].get("symbol", ""),
                "relevance": doc["similarity"],
                "data_type": chunk["metadata"].get("data_type", "unknown")
            })
            sources.add(chunk["metadata"].get("source", "unknown"))
        
        return {
            "original_query": query,
            "enhanced_contexts": contexts,
            "context_sources": list(sources),
            "total_contexts": len(contexts),
            "search_performed": True
        }
    
    async def add_market_context(self, 
                               symbols: List[str], 
                               collect_news: bool = True) -> Dict[str, int]:
        """Add market context for given symbols"""
        results = {"news_articles": 0, "total_documents": 0}
        
        if collect_news:
            news_count = await self.market_collector.collect_news_articles(symbols)
            results["news_articles"] = news_count
            results["total_documents"] += news_count
        
        return results
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        store_stats = await self.document_store.get_stats()
        
        return {
            "service_status": "operational",
            "vector_store": store_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global RAG service instance
rag_service = None

def get_rag_service():
    """Get global RAG service instance"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service