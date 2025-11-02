"""
Enhanced Vector Database Retriever for AI Incident Forecasting

This module provides a sophisticated vector store using ChromaDB (if available) 
or fallback to TF-IDF, incorporating existing MongoDB embeddings for enhanced retrieval.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vector database integration
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class EnhancedIncidentRetriever:
    """Enhanced vector-based incident retriever with ChromaDB or TF-IDF fallback"""
    
    def __init__(self, documents, metadata, use_chromadb=True, collection_name="ai_incidents"):
        """
        Initialize the enhanced incident retriever
        
        Args:
            documents (list): List of document strings (enhanced descriptions)
            metadata (list): List of metadata dictionaries for each document
            use_chromadb (bool): Whether to use ChromaDB (if available)
            collection_name (str): Name of the ChromaDB collection
        """
        self.documents = documents
        self.metadata = metadata
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.collection_name = collection_name
        
        if self.use_chromadb:
            self._setup_chromadb()
        else:
            self._setup_tfidf()
    
    def _setup_chromadb(self):
        """Set up ChromaDB vector store"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client(Settings(
                persist_directory="./vector_db",
                is_persistent=True
            ))
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                print(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "AI Incident embeddings"}
                )
                print(f"Created new ChromaDB collection: {self.collection_name}")
            
            # Check if we need to add documents
            existing_count = self.collection.count()
            if existing_count < len(self.documents):
                print(f"Adding {len(self.documents) - existing_count} new documents to ChromaDB...")
                self._add_documents_to_chromadb()
            else:
                print(f"ChromaDB collection already contains {existing_count} documents")
                
        except Exception as e:
            print(f"ChromaDB setup failed: {e}, falling back to TF-IDF")
            self.use_chromadb = False
            self._setup_tfidf()
    
    def _add_documents_to_chromadb(self):
        """Add documents to ChromaDB collection"""
        # Prepare data for ChromaDB
        ids = [f"incident_{i}" for i in range(len(self.documents))]
        
        # Extract existing embeddings if available
        embeddings = []
        for i, meta in enumerate(self.metadata):
            if 'embedding' in meta and 'vector' in meta['embedding']:
                embeddings.append(meta['embedding']['vector'])
            else:
                embeddings.append(None)
        
        # Prepare metadata for ChromaDB (convert to strings and simplify)
        chroma_metadata = []
        for meta in self.metadata:
            chroma_meta = {}
            for key, value in meta.items():
                if key != 'embedding':  # Skip embedding field
                    if isinstance(value, (str, int, float, bool)):
                        chroma_meta[key] = value
                    elif pd.notna(value):
                        chroma_meta[key] = str(value)
                    else:
                        chroma_meta[key] = "Unknown"
            chroma_metadata.append(chroma_meta)
        
        # Add to collection
        if any(emb is not None for emb in embeddings):
            # Use existing embeddings where available
            for i, (doc, meta, emb) in enumerate(zip(self.documents, chroma_metadata, embeddings)):
                try:
                    if emb is not None:
                        self.collection.add(
                            ids=[ids[i]],
                            documents=[doc],
                            metadatas=[meta],
                            embeddings=[emb]
                        )
                    else:
                        # Let ChromaDB generate embedding
                        self.collection.add(
                            ids=[ids[i]],
                            documents=[doc],
                            metadatas=[meta]
                        )
                except Exception as e:
                    print(f"Error adding document {i}: {e}")
        else:
            # Let ChromaDB generate all embeddings
            self.collection.add(
                ids=ids,
                documents=self.documents,
                metadatas=chroma_metadata
            )
    
    def _setup_tfidf(self):
        """Set up TF-IDF fallback"""
        print("Setting up TF-IDF vector store...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        print(f"TF-IDF vector store created with {len(self.documents)} documents")
    
    def retrieve(self, query, top_k=5, risk_area=None, region=None, year_range=None):
        """
        Retrieve most relevant incidents
        
        Args:
            query (str): Query text
            top_k (int): Number of results to return
            risk_area (str): Filter by risk area/domain
            region (str): Filter by geographic region
            year_range (tuple): Filter by year range (start_year, end_year)
        
        Returns:
            list: List of result dictionaries with document, metadata, and similarity score
        """
        if self.use_chromadb:
            return self._retrieve_chromadb(query, top_k, risk_area, region, year_range)
        else:
            return self._retrieve_tfidf(query, top_k, risk_area, region, year_range)
    
    def _retrieve_chromadb(self, query, top_k, risk_area, region, year_range):
        """Retrieve using ChromaDB with corrected syntax"""
        try:
            # Simple query without where clause first, then filter results
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 3, self.collection.count())  # Get more to allow for filtering
            )
            
            # Manual filtering of results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                # Apply filters manually
                skip = False
                
                if risk_area:
                    risk_domain = str(metadata.get('Risk Domain', '')).lower()
                    if risk_area.lower() not in risk_domain:
                        skip = True
                
                if region and region != "Global":
                    result_region = str(metadata.get('region', '')).lower()
                    if region.lower() not in result_region:
                        skip = True
                
                if year_range:
                    year = metadata.get('year')
                    if year and (year < year_range[0] or year > year_range[1]):
                        skip = True
                
                if not skip:
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': metadata,
                        'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
                
                if len(formatted_results) >= top_k:
                    break
            
            return formatted_results
            
        except Exception as e:
            print(f"ChromaDB query failed: {e}")
            return []
    
    def _retrieve_tfidf(self, query, top_k, risk_area, region, year_range):
        """Retrieve using TF-IDF (fallback)"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Apply filters
        valid_indices = list(range(len(self.documents)))
        
        if risk_area:
            valid_indices = [
                i for i in valid_indices 
                if risk_area.lower() in str(self.metadata[i].get('Risk Domain', '')).lower()
            ]
        
        if region and region != "Global":
            valid_indices = [
                i for i in valid_indices 
                if region.lower() in str(self.metadata[i].get('region', '')).lower()
            ]
        
        if year_range:
            start_year, end_year = year_range
            valid_indices = [
                i for i in valid_indices 
                if start_year <= self.metadata[i].get('year', 0) <= end_year
            ]
        
        # Filter similarities and get top k
        filtered_similarities = [(i, similarities[i]) for i in valid_indices]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, score in filtered_similarities[:top_k]:
            results.append({
                'document': self.documents[i],
                'metadata': self.metadata[i],
                'similarity_score': score
            })
        
        return results


def save_retriever(retriever, filepath):
    """
    Save retriever to disk
    
    Args:
        retriever (EnhancedIncidentRetriever): The retriever instance to save
        filepath (str): Path to save the retriever
    """
    retriever_data = {
        'documents': retriever.documents,
        'metadata': retriever.metadata,
        'use_chromadb': retriever.use_chromadb
    }
    if not retriever.use_chromadb:
        retriever_data['vectorizer'] = retriever.vectorizer
        retriever_data['document_vectors'] = retriever.document_vectors
    
    with open(filepath, 'wb') as f:
        pickle.dump(retriever_data, f)


def load_retriever(filepath):
    """
    Load retriever from disk
    
    Args:
        filepath (str): Path to load the retriever from
    
    Returns:
        EnhancedIncidentRetriever: Loaded retriever instance
    """
    with open(filepath, 'rb') as f:
        retriever_data = pickle.load(f)
    
    retriever = EnhancedIncidentRetriever(
        retriever_data['documents'],
        retriever_data['metadata'],
        use_chromadb=retriever_data['use_chromadb']
    )
    
    if not retriever.use_chromadb:
        retriever.vectorizer = retriever_data['vectorizer']
        retriever.document_vectors = retriever_data['document_vectors']
    
    return retriever
