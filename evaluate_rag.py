# run_evaluation.py
import os
import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import your existing RAG components
# Assuming these libraries are installed:
# pip install langchain langchain-community langchain-core faiss-cpu sentence-transformers ollama rouge_score scikit-learn
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# --- For evaluation ---
# from langchain.evaluation import load_evaluator # Optional: Requires specific setup/API keys
from langchain.schema import Document
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RAGEvalMetrics:
    """Metrics for RAG evaluation"""
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    answer_relevance: float = 0.0 # Semantic similarity to ground truth
    answer_correctness: float = 0.0 # Placeholder, potentially LLM-based or proxy
    answer_completeness: float = 0.0 # Placeholder
    rouge_l_f1: float = 0.0 # ROUGE-L F1 score for answer text
    latency_seconds: float = 0.0
    source_count: int = 0 # Number of source documents retrieved

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
        # This often requires API keys (e.g., OpenAI, Anthropic) and specific models
        # Leaving it commented out unless you configure it.
        self.llm_evaluator = None
        # try:
        #     # Example: requires OpenAI API key
        #     # self.llm_evaluator = load_evaluator("qa")
        #     pass # Placeholder if no LLM evaluator is being used
        # except Exception as e:
        #     self.llm_evaluator = None
        #     print(f"LLM evaluator not initialized: {e} - will use simpler metrics")


    def evaluate_single_query(self, query: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query against ground truth and return detailed results.

        Args:
            query: User query string
            ground_truth: Dictionary containing:
                - 'answer': str - The correct answer (required for answer metrics)
                - 'relevant_docs': List[str] - List of document IDs that should be retrieved (required for retrieval metrics)

        Returns:
            Dictionary containing evaluation results, RAG answer, and retrieved docs info.
        """
        metrics = RAGEvalMetrics()
        rag_answer = ""
        retrieved_docs_info = [] # To store info about retrieved docs (e.g., their IDs)

        # Measure latency
        start_time = datetime.now()

        try:
            # Run the query through the RAG chain
            response = self.rag_chain.invoke({"query": query})

            # Get the generated answer and retrieved documents
            rag_answer = response.get('result', '')
            retrieved_docs = response.get('source_documents', [])
            metrics.source_count = len(retrieved_docs)

            # Collect info about retrieved documents for later inspection
            for doc in retrieved_docs:
                doc_id = doc.metadata.get('doc_id', doc.metadata.get('source', 'unknown_doc_id')) # Prioritize doc_id from your indexer
                retrieved_docs_info.append({'doc_id': doc_id, 'source': doc.metadata.get('source'), 'page': doc.metadata.get('page'), 'chunk_snippet': doc.page_content[:100] + '...'})


            # Calculate latency
            end_time = datetime.now()
            metrics.latency_seconds = (end_time - start_time).total_seconds()

            # 1. Evaluate retrieval quality (only if ground truth docs are provided)
            ground_truth_relevant_docs = ground_truth.get('relevant_docs')
            if ground_truth_relevant_docs is not None:
                 if retrieved_docs or ground_truth_relevant_docs: # Avoid division by zero if both are empty
                    metrics.retrieval_precision, metrics.retrieval_recall = self._evaluate_retrieval(
                        retrieved_docs,
                        ground_truth_relevant_docs
                    )
                 else: # If both retrieved and relevant are empty, perfect precision/recall (depends on interpretation, but 1.0 is reasonable if the query requires no docs)
                     metrics.retrieval_precision = 1.0
                     metrics.retrieval_recall = 1.0
            else:
                 # Set to None or some indicator if ground truth isn't available for this metric
                 metrics.retrieval_precision = None
                 metrics.retrieval_recall = None


            # 2. Evaluate answer quality (only if ground truth answer is provided)
            ground_truth_answer = ground_truth.get('answer')
            if ground_truth_answer is not None:
                # Calculate ROUGE-L score
                # Need to handle cases where answer or ground_truth_answer is empty
                if rag_answer and ground_truth_answer:
                     rouge_scores = self.rouge_scorer.score(rag_answer, ground_truth_answer)
                     metrics.rouge_l_f1 = rouge_scores['rougeL'].fmeasure
                else:
                     metrics.rouge_l_f1 = 0.0 # No overlap if one text is empty

                # Calculate semantic similarity
                # Need to handle cases where answer or ground_truth_answer is empty
                if rag_answer and ground_truth_answer:
                     metrics.answer_relevance = self._calculate_semantic_similarity(
                         rag_answer, ground_truth_answer
                     )
                else:
                     metrics.answer_relevance = 0.0 # No similarity if one text is empty


                # Use LLM evaluator if available (requires specific setup, often API keys)
                # This part is commented out unless you configure a LangChain LLM evaluator
                # if self.llm_evaluator:
                #     try:
                #         # Note: The exact evaluation logic depends on the specific evaluator (e.g., 'qa')
                #         # It might require the context used to generate the answer.
                #         # The current setup doesn't easily pass the context used BY THE CHAIN
                #         # to the evaluator here. This is a limitation of this simple structure.
                #         # For accurate LLM evaluation, you might need to capture the context
                #         # returned by the chain's invoke call.
                #         eval_result = self.llm_evaluator.evaluate_strings(
                #             prediction=rag_answer,
                #             reference=ground_truth_answer,
                #             question=query
                #             # Optionally, context if available: context="\n\n".join([d.page_content for d in retrieved_docs])
                #         )
                #         # Depending on the evaluator, scores might be different types or keys
                #         metrics.answer_correctness = float(eval_result.get('score', 0.0)) # Adjust key if needed
                #         # metrics.answer_completeness = ... # If evaluator provides it
                #     except Exception as e:
                #          print(f"Error during LLM evaluation for query '{query[:50]}...': {e}")
                #          metrics.answer_correctness = 0.0 # Mark as failed
                # else: # If no LLM evaluator, use semantic similarity as a simple proxy for correctness
                     metrics.answer_correctness = metrics.answer_relevance # Using semantic similarity as a simple proxy
            else:
                 # Set to None or some indicator if ground truth isn't available for these metrics
                 metrics.rouge_l_f1 = None
                 metrics.answer_relevance = None
                 metrics.answer_correctness = None # Or set to None as well


        except Exception as e:
            print(f"Error processing query '{query[:50]}...': {e}")
            # Log error and return metrics with defaults/error states
            metrics.latency_seconds = (datetime.now() - start_time).total_seconds() # Still capture how long it took to fail
            rag_answer = f"Error processing query: {e}" # Indicate failure in answer
            metrics.retrieval_precision = metrics.retrieval_recall = metrics.answer_relevance = metrics.answer_correctness = metrics.rouge_l_f1 = 0.0 # Set to 0 or None on error


        return {
            'query': query,
            'ground_truth_answer': ground_truth.get('answer'), # Use .get() to avoid KeyError if key is missing
            'ground_truth_relevant_docs': ground_truth.get('relevant_docs'), # Use .get()
            'rag_answer': rag_answer,
            'retrieved_docs_info': retrieved_docs_info, # Store info about retrieved docs
            'metrics': metrics.to_dict()
        }

    def _evaluate_retrieval(self, retrieved_docs: List[Document],
                           relevant_doc_ids: List[str]) -> tuple:
        """
        Calculate precision and recall for document retrieval
        Checks for 'doc_id' in metadata first, which matches your indexing script.

        Args:
            retrieved_docs: List of retrieved Document objects (chunks).
                            These should have 'doc_id' in their metadata.
            relevant_doc_ids: List of string IDs of the original documents
                              that should have had chunks retrieved.

        Returns:
            Tuple of (precision, recall)
        """
        # Extract original document IDs from retrieved chunks
        # Use a set for retrieved IDs to handle multiple chunks from the same source document
        retrieved_original_doc_ids_set = set()
        for doc in retrieved_docs:
            # This logic must match how your indexer stores the original ID
            doc_id = doc.metadata.get('doc_id')
            if doc_id is not None:
                 retrieved_original_doc_ids_set.add(str(doc_id).strip().lower())
            # else:
                 # print(f"Warning: Retrieved document missing 'doc_id' metadata: {doc.metadata}") # Optional: Debug missing IDs


        # Normalize ground truth IDs for comparison
        normalized_relevant_doc_ids_set = set(str(id).strip().lower() for id in relevant_doc_ids if id is not None)

        # Calculate precision and recall
        # Precision: proportion of retrieved documents that are relevant
        # Relevant retrieved = Intersection of retrieved and relevant sets
        correct_retrievals = retrieved_original_doc_ids_set.intersection(normalized_relevant_doc_ids_set)

        # Denominators
        total_retrieved = len(retrieved_original_doc_ids_set)
        total_relevant = len(normalized_relevant_doc_ids_set)

        precision = len(correct_retrievals) / total_retrieved if total_retrieved > 0 else (1.0 if total_relevant == 0 else 0.0) # Perfect precision if nothing retrieved and nothing relevant
        recall = len(correct_retrievals) / total_relevant if total_relevant > 0 else (1.0 if total_retrieved == 0 else 0.0) # Perfect recall if nothing relevant and nothing retrieved

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
        try:
            # Handle empty strings gracefully
            if not text1 or not text2:
                return 0.0

            # Get embeddings
            embedding1 = self.embedding_model.embed_query(text1)
            embedding2 = self.embedding_model.embed_query(text2)

            # Reshape for cosine_similarity function (requires 2D arrays)
            embedding1 = np.asarray(embedding1).reshape(1, -1)
            embedding2 = np.asarray(embedding2).reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding1,
                embedding2
            )[0][0]

            # Cosine similarity can sometimes be slightly outside [ -1, 1] due to floating point errors
            return max(0.0, float(similarity)) # Ensure score is non-negative

        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0


    def evaluate_test_set(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a full test dataset

        Args:
            test_data: List of dictionaries, each containing:
                - 'query': str - The user query
                - 'answer': str - The correct answer (optional, for answer metrics)
                - 'relevant_docs': List[str] - List of document IDs that should be retrieved (optional, for retrieval metrics)

        Returns:
            Dictionary with aggregate metrics and individual query results
        """
        individual_results = []

        # Process each test query
        for i, test_item in enumerate(test_data):
            query = test_item.get('query', f'Query_{i+1}') # Handle potential missing query
            # Ensure required ground truth keys are present, even if their values are empty lists/strings/None
            ground_truth = {
                'answer': test_item.get('answer', None), # Use None if key is missing
                'relevant_docs': test_item.get('relevant_docs', None) # Use None if key is missing
            }

            print(f"Evaluating query {i+1}/{len(test_data)}: {query[:70]}...") # Print more of the query

            # Evaluate the query and get detailed result
            query_result = self.evaluate_single_query(
                query,
                ground_truth
            )

            # Store the full result (includes query, answers, docs, metrics)
            individual_results.append(query_result)

        # Calculate aggregate metrics from the collected individual results
        agg_metrics = self._calculate_aggregate_metrics(individual_results)

        return {
            'aggregate_metrics': agg_metrics,
            'individual_results': individual_results
        }

    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results"""
        metrics = {}

        # Metrics to aggregate - ensure these match RAGEvalMetrics fields
        metric_keys = [
            'retrieval_precision', 'retrieval_recall',
            'answer_relevance', 'answer_correctness', 'answer_completeness', # answer_completeness is currently a placeholder
            'rouge_l_f1', 'latency_seconds', 'source_count'
        ]

        # Initialize lists for averaging, only for metrics that were actually calculated (not None)
        metric_values = {key: [] for key in metric_keys}

        # Collect metrics
        for result in results:
            result_metrics = result['metrics']
            for key in metric_keys:
                 value = result_metrics.get(key)
                 # Only include non-None values in the aggregation
                 if value is not None:
                    metric_values[key].append(value)

        # Calculate averages
        agg_metrics = {}
        for key in metric_keys:
            # Calculate average only if there are values, otherwise default to 0.0 or indicate N/A
            agg_metrics[key] = sum(metric_values[key]) / len(metric_values[key]) if metric_values[key] else 0.0 # Use 0.0 or maybe None? Let's stick to 0.0 for now.

        return agg_metrics


# Removed create_test_set as we are defining test_data directly

def run_evaluation(faiss_index_dir, embedding_model_name, ollama_model_name):
    """Run a full RAG evaluation"""
    print("Initializing RAG components for evaluation...")

    # Load embedding model
    try:
        embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"Embedding model '{embedding_model_name}' initialized.")
    except Exception as e:
        print(f"Error initializing embedding model '{embedding_model_name}': {e}")
        print("Please ensure the model files are available locally or accessible.")
        return None # Exit if embedding model fails

    # Load FAISS index
    try:
        vectorstore = FAISS.load_local(
            faiss_index_dir,
            embedding,
            allow_dangerous_deserialization=True # Be cautious with this in production or untrusted sources
        )
        print(f"FAISS index loaded successfully from {faiss_index_dir}")
    except Exception as e:
        print(f"Error loading FAISS index from {faiss_index_dir}: {e}")
        print("Please ensure the index exists at the specified path and matches the embedding model.")
        return None # Exit if index loading fails

    # Initialize LLM
    try:
        print(f"Initializing Ollama model: {ollama_model_name}...")
        llm = Ollama(model=ollama_model_name)
        # Basic test call (optional but helpful for debugging)
        try:
            llm.invoke("Hello", stop=["\n"], max_tokens=1) # Send a minimal request
            print(f"Ollama model '{ollama_model_name}' appears to be running.")
        except Exception as llm_invoke_e:
            print(f"Warning: Could not confirm Ollama '{ollama_model_name}' is fully responsive: {llm_invoke_e}")

    except Exception as e:
         print(f"Error initializing Ollama model '{ollama_model_name}': {e}")
         print("Please ensure Ollama is running and the model is available ('ollama list').")
         return None # Exit if LLM initialization fails


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
        return_source_documents=True # Crucial for evaluation to get the retrieved docs
    )

    # Create evaluator
    evaluator = RAGEvaluator(rag_chain, embedding_model=embedding)

    # --- Define the Test Data based on the provided scenarios ---
    # This is human-defined ground truth for evaluation.
    # IMPORTANT: The 'relevant_docs' IDs MUST match the 'id' field from your
    # original JSONL files for these specific scenarios, as your indexer
    # puts that 'id' into the 'doc_id' metadata field.
    # Replace the placeholder IDs below with your actual document IDs from your JSONL data.
    test_data = [
        {
            'query': "My business details were stolen through a phishing attack disguised as a government email, leading to fraudulent loans. What Indian laws are relevant and what steps should I take?",
            'answer': "If your business details were stolen via a phishing attack and used for fraudulent loans, relevant Indian laws include Section 66D (punishment for cheating by personation by using computer resource) and Section 66C (punishment for identity theft) of the IT Act, 2000. Additionally, sections related to cheating (e.g., Section 420) and forgery under the Bharatiya Nyaya Sanhita (BNS) or Indian Penal Code (IPC) may apply. You should immediately report the incident to the Cyber Crime Cell, gather all evidence like the email headers and fake website URL, and file a formal police complaint.",
            'relevant_docs': [
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66D', # Replace with actual ID from your data
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66C', # Replace with actual ID from your data
                'YOUR_JSONL_ID_FOR_IPC_BNS_CHEATING',  # Replace with actual ID (e.g., document about IPC 420 or BNS equivalent)
                'YOUR_JSONL_ID_FOR_IPC_BNS_FORGERY',   # Replace with actual ID (e.g., document about IPC Forgery sections)
                'YOUR_JSONL_ID_FOR_CYBER_CRIME_REPORT_PROCEDURE', # Replace with actual ID for a document describing this procedure
                'YOUR_JSONL_ID_FOR_POLICE_COMPLAINT_GUIDE' # Replace with actual ID for a document describing this procedure
            ]
        },
        {
            'query': "My data from an online shopping portal was breached, and I received a fraudulent SMS asking for card details for a refund. What laws apply to the portal's responsibility and the scammer?",
            'answer': "In a data breach scenario from an online portal followed by a refund scam attempt, the scammer's actions primarily fall under the IT Act, 2000 (Section 66 - computer related offenses, Section 66D - cheating by personation) and IPC/BNS (cheating). The online portal's potential liability for the breach itself could involve IT Act Section 43A (compensation for failure to protect data) or Section 72A (punishment for disclosure of information in breach of lawful contract), depending on their negligence in protecting data. The Consumer Protection Act, 2019, might also apply regarding deficiency in service or unfair trade practices by the platform. You should report to the online platform, your bank, and file a complaint with the Cyber Crime Cell. Consider consumer court action if the platform was negligent.",
            'relevant_docs': [
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_43A',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_72A',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66D',
                'YOUR_JSONL_ID_FOR_IPC_BNS_CHEATING',
                'YOUR_JSONL_ID_FOR_CONSUMER_PROTECTION_ACT_2019',
                'YOUR_JSONL_ID_FOR_PROCEDURE_REPORT_BREACH_PLATFORM',
                'YOUR_JSONL_ID_FOR_PROCEDURE_REPORT_BANK',
                'YOUR_JSONL_ID_FOR_CYBER_CRIME_REPORT_PROCEDURE',
                'YOUR_JSONL_ID_FOR_CONSUMER_COURT_PROCEDURE'
            ]
        },
         {
            'query': "My ex is cyberstalking me, sending threats, spreading rumors online, and threatening to share private photos. What Indian laws cover this and what should I do?",
            'answer': "Cyberstalking involving threats, harassment, spreading rumors, and threatening to share private photos is covered under several Indian laws. This includes potentially IT Act Section 67, 67A, or 67B (publishing/transmitting obscene/sexually explicit material) if photos are involved, and potentially Section 66 (computer-related offenses). More significantly, sections from the Bharatiya Nyaya Sanhita (BNS) or Indian Penal Code (IPC) apply, such as those related to criminal intimidation (e.g., Section 506 IPC/BNS), stalking (specific IPC/BNS provisions introduced), and defamation (Section 499/500 IPC, or Section 356 BNS) for spreading rumors. To address this, collect all evidence (screenshots, message logs), block the harasser, report the content/user to social media platforms, and file a complaint with the Cyber Crime Cell.",
            'relevant_docs': [
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_67',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_67A',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_67B',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66',
                'YOUR_JSONL_ID_FOR_IPC_BNS_CRIMINAL_INTIMIDATION',
                'YOUR_JSONL_ID_FOR_IPC_BNS_STALKING_PROVISIONS',
                'YOUR_JSONL_ID_FOR_IPC_BNS_DEFAMATION',
                'YOUR_JSONL_ID_FOR_PROCEDURE_COLLECT_DIGITAL_EVIDENCE',
                'YOUR_JSONL_ID_FOR_PROCEDURE_REPORT_SOCIAL_MEDIA',
                'YOUR_JSONL_ID_FOR_CYBER_CRIME_REPORT_PROCEDURE'
            ]
        },
        {
            'query': "Money was stolen from my e-wallet after using public Wi-Fi, authenticated by an OTP I didn't provide. What legal sections under the IT Act cover this misuse?",
            'answer': "If money was stolen from your e-wallet authenticated by an unauthorized OTP after your system was potentially compromised on public Wi-Fi, relevant sections under the IT Act, 2000 include Section 43 (Penalty and compensation for damage to computer system), Section 66 (Computer related offenses), Section 66C (Punishment for identity theft – specifically related to password, digital signature, or other unique identification feature like an OTP), and potentially Section 66B (Punishment for dishonestly receiving stolen computer resource or communication device). You must immediately contact your e-wallet provider/bank to block accounts and attempt to reverse the transaction, then report the incident to the Cyber Crime Cell, gathering any technical evidence if possible (e.g., IP logs from the cafe if available).",
            'relevant_docs': [
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_43',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66C',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66B',
                'YOUR_JSONL_ID_FOR_PROCEDURE_CONTACT_EWALLET_BANK',
                'YOUR_JSONL_ID_FOR_CYBER_CRIME_REPORT_PROCEDURE'
            ]
        },
        {
            'query': "A travel agency used my original photographs from my website on their commercial site and social media without my permission. What Indian laws protect my work and what steps can I take?",
            'answer': "Unauthorized use of your original photographs, which are considered digital content, is primarily covered by the Copyright Act, 1957. This Act protects your intellectual property rights as the creator. While the Copyright Act is the main law for infringement, the IT Act, 2000 could potentially be relevant under Section 43 (damage) if the act of unauthorized copying or distribution caused issues to your system or data, or general sections related to offenses using a computer resource. To address this, you should send a formal legal notice (cease and desist letter) to the travel agency, gather clear evidence of the infringement (screenshots of their site/posts, links), and consider filing a civil lawsuit for injunction and damages under the Copyright Act. A criminal complaint is also possible if there's clear intent to infringe and profit illegally.",
            'relevant_docs': [
                'YOUR_JSONL_ID_FOR_COPYRIGHT_ACT_1957',
                'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_43', # Less primary, include if document discusses digital IP theft using computers
                'YOUR_JSONL_ID_FOR_PROCEDURE_SEND_LEGAL_NOTICE',
                'YOUR_JSONL_ID_FOR_PROCEDURE_GATHER_IP_EVIDENCE',
                'YOUR_JSONL_ID_FOR_PROCEDURE_FILE_CIVIL_SUIT_COPYRIGHT',
                'YOUR_JSONL_ID_FOR_PROCEDURE_FILE_CRIMINAL_COMPLAINT_IP' # Optional: Include if document covers this route
            ]
        }
        # Add more test cases here following the same structure
    ]

    print(f"Prepared {len(test_data)} test scenarios with human-defined ground truth.")

    # Run evaluation
    print("Running evaluation...")
    eval_results = evaluator.evaluate_test_set(test_data)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"rag_evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Use a custom encoder or ensure all data is serializable (dataclasses handle this with asdict)
        json.dump(eval_results, f, indent=2)

    print(f"Evaluation complete! Results saved to {results_file}")

    # Print summary
    print("\n=== RAG Evaluation Summary ===")
    if 'aggregate_metrics' in eval_results:
        for metric, value in eval_results['aggregate_metrics'].items():
            # Format percentages, handle None if metric wasn't calculated for any query
            if isinstance(value, float) and 0.0 <= value <= 1.0:
                 print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            elif metric == 'latency_seconds':
                 print(f"{metric.replace('_', ' ').title()}: {value:.2f}s")
            elif metric == 'source_count':
                 print(f"{metric.replace('_', ' ').title()}: {value:.1f}")
            elif value is None:
                 print(f"{metric.replace('_', ' ').title()}: N/A")
            else:
                 print(f"{metric.replace('_', ' ').title()}: {value}")

    else:
        print("Could not calculate aggregate metrics.")

    return eval_results


if __name__ == "__main__":
    # Configuration - update with your paths and model names
    FAISS_INDEX_DIR = "./faiss_index_legal" # Must match your indexing script
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Must match your indexing script
    OLLAMA_MODEL_NAME = "mistral" # Ensure this model is downloaded and running in Ollama

    # --- IMPORTANT ---
    # Before running:
    # 1. Make sure you have run your indexing script (the one you provided)
    #    to create the FAISS index at FAISS_INDEX_DIR.
    # 2. Make sure Ollama is running and the OLLAMA_MODEL_NAME model is available.
    # 3. Update the 'relevant_docs' lists in the 'test_data' above
    #    with the actual 'id' values from your JSONL source data.
    #    e.g., replace 'YOUR_JSONL_ID_FOR_IT_ACT_SECTION_66D' with something like
    #    'indian_laws_it_act_2000_section_66d' if that's the 'id' in your JSONL.
    #    If you don't update these, the retrieval metrics will be incorrect.
    # ------------------


    # Run evaluation
    print("Starting RAG evaluation...")
    results = run_evaluation(FAISS_INDEX_DIR, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME)

    if results:
        print("\nEvaluation run complete. You can now run the analysis script.")
    else:
        print("\nEvaluation failed during setup or execution.")