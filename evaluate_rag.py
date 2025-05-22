import os
import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import your existing RAG components
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# --- For evaluation ---
from langchain.evaluation import load_evaluator
from langchain.schema import Document
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RAGEvalMetrics:
    """Metrics for RAG evaluation"""
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    answer_completeness: float = 0.0
    rouge_l_f1: float = 0.0
    latency_seconds: float = 0.0
    source_count: int = 0
    
    def to_dict(self):
        return asdict(self)

class RAGEvaluator:
    """Class for evaluating RAG system performance"""
    
    def __init__(self, rag_chain: RetrievalQA, embedding_model=None):
        self.rag_chain = rag_chain
        # Use the same embedding model as the RAG system if provided
        # Otherwise, initialize a new one
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # For text similarity metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Optional: Initialize LangChain evaluator if you want to use it
        try:
            # Only works if you have OpenAI API key set
            # self.llm_evaluator = load_evaluator("qa")
            self.llm_evaluator = None
        except:
            self.llm_evaluator = None
            print("LLM evaluator not initialized - will use simpler metrics")
    
    def evaluate_single_query(self, query: str, ground_truth: Dict[str, Any]) -> RAGEvalMetrics:
        """
        Evaluate a single query against ground truth
        
        Args:
            query: User query string
            ground_truth: Dictionary containing:
                - 'answer': str - The correct answer
                - 'relevant_docs': List[str] - List of document IDs that should be retrieved
        
        Returns:
            RAGEvalMetrics object with evaluation results
        """
        metrics = RAGEvalMetrics()
        
        # Measure latency
        start_time = datetime.now()
        
        # Run the query through the RAG chain
        response = self.rag_chain.invoke({"query": query})
        
        # Calculate latency
        end_time = datetime.now()
        metrics.latency_seconds = (end_time - start_time).total_seconds()
        
        # Get the generated answer and retrieved documents
        answer = response.get('result', '')
        retrieved_docs = response.get('source_documents', [])
        metrics.source_count = len(retrieved_docs)
        
        # 1. Evaluate retrieval quality
        if ground_truth.get('relevant_docs') and retrieved_docs:
            metrics.retrieval_precision, metrics.retrieval_recall = self._evaluate_retrieval(
                retrieved_docs, 
                ground_truth.get('relevant_docs', [])
            )
        
        # 2. Evaluate answer quality
        if ground_truth.get('answer'):
            # Calculate ROUGE-L score
            rouge_scores = self.rouge_scorer.score(answer, ground_truth['answer'])
            metrics.rouge_l_f1 = rouge_scores['rougeL'].fmeasure
            
            # Calculate semantic similarity
            metrics.answer_relevance = self._calculate_semantic_similarity(
                answer, ground_truth['answer']
            )
            
            # Use LLM evaluator if available
            if self.llm_evaluator:
                correctness = self.llm_evaluator.evaluate_strings(
                    prediction=answer,
                    reference=ground_truth['answer'],
                    question=query
                )
                metrics.answer_correctness = correctness.get('score', 0.0)
            else:
                # Default to semantic similarity as a proxy for correctness
                metrics.answer_correctness = metrics.answer_relevance
        
        return metrics
    
    def _evaluate_retrieval(self, retrieved_docs: List[Document], 
                           relevant_doc_ids: List[str]) -> tuple:
        """
        Calculate precision and recall for document retrieval
        
        Args:
            retrieved_docs: List of retrieved Document objects
            relevant_doc_ids: List of relevant document IDs that should have been retrieved
            
        Returns:
            Tuple of (precision, recall)
        """
        # Extract document IDs from retrieved documents
        retrieved_ids = []
        for doc in retrieved_docs:
            # Try to get doc_id from metadata in several possible locations
            doc_id = doc.metadata.get('doc_id')
            if not doc_id:
                # Fallback - construct ID from source and page if available
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', '')
                doc_id = f"{source}_{page}" if page else source
            retrieved_ids.append(doc_id)
        
        # Find the intersection between retrieved and relevant
        correct_retrievals = set(retrieved_ids).intersection(set(relevant_doc_ids))
        
        # Calculate precision and recall
        precision = len(correct_retrievals) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(correct_retrievals) / len(relevant_doc_ids) if relevant_doc_ids else 0
        
        return precision, recall
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Get embeddings
        embedding1 = self.embedding_model.embed_query(text1)
        embedding2 = self.embedding_model.embed_query(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            [embedding1], 
            [embedding2]
        )[0][0]
        
        return float(similarity)

    def evaluate_test_set(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a full test dataset
        
        Args:
            test_data: List of dictionaries, each containing:
                - 'query': str - The user query
                - 'answer': str - The correct answer
                - 'relevant_docs': List[str] - List of document IDs that should be retrieved
                
        Returns:
            Dictionary with aggregate metrics and individual query results
        """
        results = []
        
        # Process each test query
        for i, test_item in enumerate(test_data):
            query = test_item['query']
            print(f"Evaluating query {i+1}/{len(test_data)}: {query[:50]}...")
            
            # Evaluate the query
            metrics = self.evaluate_single_query(
                query, 
                {
                    'answer': test_item.get('answer', ''),
                    'relevant_docs': test_item.get('relevant_docs', [])
                }
            )
            
            # Store results
            results.append({
                'query': query,
                'metrics': metrics.to_dict()
            })
            
        # Calculate aggregate metrics
        agg_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'aggregate_metrics': agg_metrics,
            'individual_results': results
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results"""
        metrics = {}
        
        # Metrics to aggregate
        metric_keys = [
            'retrieval_precision', 'retrieval_recall', 
            'answer_relevance', 'answer_correctness', 'answer_completeness',
            'rouge_l_f1', 'latency_seconds', 'source_count'
        ]
        
        # Initialize metrics
        for key in metric_keys:
            metrics[key] = 0.0
            
        # Sum metrics
        for result in results:
            result_metrics = result['metrics']
            for key in metric_keys:
                metrics[key] += result_metrics.get(key, 0.0)
        
        # Calculate averages
        for key in metric_keys:
            metrics[key] = metrics[key] / len(results) if results else 0.0
            
        return metrics


def create_test_set(queries, vectorstore=None):
    """
    Helper function to create a test set with automatically generated ground truth answers
    using the same RAG system (useful when you don't have human-labeled data)
    
    Args:
        queries: List of test queries
        vectorstore: Optional FAISS index to use for finding relevant docs
    
    Returns:
        List of test cases with queries and auto-generated "ground truth"
    """
    # Initialize a larger context model for generating reference answers
    # Ideally this would be a stronger model than your RAG system uses
    reference_llm = Ollama(model="mistral") # Consider using a larger model
    
    test_data = []
    
    for query in queries:
        test_case = {'query': query}
        
        # If vectorstore provided, find relevant documents
        if vectorstore:
            # Get top 5 documents for the query
            docs = vectorstore.similarity_search(query, k=5)
            # Extract doc IDs - these become our "ground truth" for retrieval
            doc_ids = []
            for doc in docs:
                doc_id = doc.metadata.get('doc_id')
                if not doc_id:
                    source = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', '')
                    doc_id = f"{source}_{page}" if page else source
                doc_ids.append(doc_id)
            test_case['relevant_docs'] = doc_ids
            
            # Generate a reference answer using these docs
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""
            Based on the following context information, provide a comprehensive answer to the question.
            
            Context:
            {context}
            
            Question:
            {query}
            
            Answer:
            """
            test_case['answer'] = reference_llm.invoke(prompt)
        
        test_data.append(test_case)
    
    return test_data


def run_evaluation(faiss_index_dir, embedding_model_name, ollama_model_name):
    """Run a full RAG evaluation"""
    print("Initializing RAG components for evaluation...")
    
    # Load embedding model
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Load FAISS index
    vectorstore = FAISS.load_local(
        faiss_index_dir,
        embedding,
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM
    llm = Ollama(model=ollama_model_name)
    
    # Set up the prompt template (use the same as your application)
    prompt_template = """You are a professional legal assistant powered by AI. Your role is to:
    1. Provide clear explanations of legal concepts and procedures
    2. Reference relevant laws, regulations, and precedents from the provided context
    3. Help users understand their legal rights and obligations
    4. Explain legal terminology in plain language

    Important Guidelines:
    - Always base responses on the provided context documents
    - Cite specific sources, including document names and page numbers when available
    - Clearly state when information is not available in the context
    - Maintain a professional, clear, and objective tone

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    # Create evaluator
    evaluator = RAGEvaluator(rag_chain, embedding_model=embedding)
    
    # OPTION 1: Use domain-specific test queries that users might ask
    test_queries = [
        """Mrs. Devi, a small business owner, receives an email supposedly from a 
government department (e.g., GST office) asking her to update her business registration 
details by clicking a link. The link leads to a professional-looking but fake website. 
Believing it's genuine, she enters her business registration number, PAN, and other 
sensitive details. Later, she finds that someone has used these details to apply for 
fraudulent loans in her business's name. """,
"""Mr. Khan uses a popular online shopping portal. One day, he receives an 
SMS from an unknown number stating that his recent order has been cancelled and 
asks him to call a specific number for a refund. When he calls, the person on the other 
end asks for his debit card details, claiming it's for the refund process. He suspects his 
data from the shopping portal might have been compromised.""",
""" Mr. Rao, a freelancer, uses a public Wi-Fi network at a cafe. Unbeknownst to 
him, his system is compromised. Later, he discovers that a significant amount of money 
has been transferred from his e-wallet to an unknown account, and he receives an SMS 
notification for an OTP he never initiated or shared. The transaction was digitally signed 
or authenticated using an OTP he didn't provide.""",
""" A budding photographer, Aisha, uploads her original landscape photographs 
to her personal website. She later discovers that a well-known travel agency has 
downloaded her photos and is using them on their commercial website and social media 
without her permission or credit, essentially claiming them as their own."""
        # Add more domain-specific queries here
    ]
    
    # Create test set with auto-generated "ground truth"
    print("Creating test dataset...")
    test_data = create_test_set(test_queries, vectorstore)
    
    # OPTION 2: If you have human-labeled test data, load it here instead
    # with open('test_data.json', 'r') as f:
    #     test_data = json.load(f)
    
    # Run evaluation
    print("Running evaluation...")
    eval_results = evaluator.evaluate_test_set(test_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"rag_evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation complete! Results saved to {results_file}")
    
    # Print summary
    print("\n=== RAG Evaluation Summary ===")
    for metric, value in eval_results['aggregate_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    return eval_results


if __name__ == "__main__":
    # Configuration - update with your paths
    FAISS_INDEX_DIR = "./faiss_index_legal"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_MODEL_NAME = "mistral"
    
    # Run evaluation
    results = run_evaluation(FAISS_INDEX_DIR, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME)