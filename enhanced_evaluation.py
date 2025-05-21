import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class RAGEvaluationAnalyzer:
    """Class for analyzing and visualizing RAG evaluation results"""
    
    def __init__(self, results_file: str = None, results_data: Dict = None):
        """
        Initialize with either a file path or results data
        
        Args:
            results_file: Path to JSON file with evaluation results
            results_data: Dictionary containing evaluation results
        """
        if results_file:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        elif results_data:
            self.results = results_data
        else:
            raise ValueError("Either results_file or results_data must be provided")
        
        # Extract metrics
        self.aggregate_metrics = self.results['aggregate_metrics']
        self.individual_results = self.results['individual_results']
        
    def generate_summary_report(self) -> str:
        """Generate a text summary report of evaluation results"""
        report = []
        
        # Header
        report.append("# RAG Evaluation Summary Report")
        report.append("\n## Overall Performance Metrics\n")
        
        # Format aggregate metrics
        metrics_table = "| Metric | Value |\n|--------|-------|\n"
        for metric, value in self.aggregate_metrics.items():
            # Format metrics nicely
            if metric == 'latency_seconds':
                formatted_value = f"{value:.2f}s"
            elif metric == 'source_count':
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.2%}"
            
            metrics_table += f"| {metric.replace('_', ' ').title()} | {formatted_value} |\n"
        
        report.append(metrics_table)
        
        # Interpretation
        report.append("\n## Interpretation\n")
        
        # Analyze retrieval quality
        retrieval_precision = self.aggregate_metrics['retrieval_precision']
        retrieval_recall = self.aggregate_metrics['retrieval_recall']
        
        if retrieval_precision > 0.8 and retrieval_recall > 0.8:
            retrieval_assessment = "Excellent retrieval performance. The system is finding most relevant documents with high precision."
        elif retrieval_precision > 0.6 and retrieval_recall > 0.6:
            retrieval_assessment = "Good retrieval performance. The system retrieves most relevant documents but with some irrelevant results."
        elif retrieval_precision > retrieval_recall:
            retrieval_assessment = "The system is precise but may be missing some relevant documents. Consider adjusting the retrieval to be more inclusive."
        elif retrieval_recall > retrieval_precision:
            retrieval_assessment = "The system finds many relevant documents but also includes too many irrelevant ones. Consider making retrieval more selective."
        else:
            retrieval_assessment = "Retrieval performance needs improvement. Consider retraining or adjusting the vector database."
            
        report.append(f"### Retrieval Quality\n{retrieval_assessment}")
        
        # Analyze answer quality
        answer_relevance = self.aggregate_metrics['answer_relevance']
        answer_correctness = self.aggregate_metrics['answer_correctness']
        rouge_score = self.aggregate_metrics['rouge_l_f1']
        
        if answer_relevance > 0.85 and rouge_score > 0.6:
            answer_assessment = "Excellent answer quality. Responses are relevant and closely match reference answers."
        elif answer_relevance > 0.7 and rouge_score > 0.4:
            answer_assessment = "Good answer quality. Responses are relevant but may miss some details from reference answers."
        elif answer_relevance > 0.6:
            answer_assessment = "Acceptable answer relevance, but answers may be missing important details or context."
        else:
            answer_assessment = "Answer quality needs improvement. Consider adjusting the prompt template or using a different LLM."
            
        report.append(f"\n### Answer Quality\n{answer_assessment}")
        
        # Analyze performance characteristics
        latency = self.aggregate_metrics['latency_seconds']
        sources = self.aggregate_metrics['source_count']
        
        if latency > 5:
            latency_assessment = f"System latency ({latency:.2f}s) is high. Consider optimizing the retrieval process or using a faster LLM."
        else:
            latency_assessment = f"System latency ({latency:.2f}s) is acceptable."
            
        report.append(f"\n### Performance Characteristics\n{latency_assessment}")
        report.append(f"The system uses an average of {sources:.1f} sources per query.")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        
        recommendations = []
        if retrieval_precision < 0.7:
            recommendations.append("- Improve retrieval precision by refining the document chunking strategy or embedding model.")
        if retrieval_recall < 0.7:
            recommendations.append("- Improve retrieval recall by including more documents in the search or adjusting the similarity threshold.")
        if answer_relevance < 0.7:
            recommendations.append("- Improve answer relevance by refining the prompt template to better guide the LLM.")
        if rouge_score < 0.5:
            recommendations.append("- Improve answer completeness by providing more context to the LLM or using a more capable model.")
        if latency > 3:
            recommendations.append("- Reduce latency by optimizing the retrieval process or using a faster LLM implementation.")
        
        if not recommendations:
            recommendations.append("- The system is performing well. Consider testing with a larger and more diverse query set.")
            
        report.append("\n".join(recommendations))
        
        return "\n".join(report)
    
    def create_visualization(self, output_file: str = "rag_evaluation_charts.png"):
        """Create visualization of evaluation metrics"""
        # Convert individual results to DataFrame for easier plotting
        rows = []
        for result in self.individual_results:
            row = {'query': result['query'][:30] + '...' if len(result['query']) > 30 else result['query']}
            row.update(result['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Set up the visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG System Evaluation Results', fontsize=16)
        
        # Plot 1: Retrieval metrics
        if 'retrieval_precision' in df.columns and 'retrieval_recall' in df.columns:
            retrieval_df = df[['query', 'retrieval_precision', 'retrieval_recall']].copy()
            retrieval_df = retrieval_df.set_index('query')
            retrieval_df.plot(kind='bar', ax=axes[0, 0], ylim=(0, 1))
            axes[0, 0].set_title('Retrieval Performance by Query')
            axes[0, 0].set_ylabel('Score (0-1)')
            axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
            axes[0, 0].legend(['Precision', 'Recall'])
        
        # Plot 2: Answer quality metrics
        if 'answer_relevance' in df.columns and 'rouge_l_f1' in df.columns:
            answer_df = df[['query', 'answer_relevance', 'rouge_l_f1']].copy()
            answer_df = answer_df.set_index('query')
            answer_df.plot(kind='bar', ax=axes[0, 1], ylim=(0, 1))
            axes[0, 1].set_title('Answer Quality by Query')
            axes[0, 1].set_ylabel('Score (0-1)')
            axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
            axes[0, 1].legend(['Relevance', 'ROUGE-L F1'])
        
        # Plot 3: Latency
        if 'latency_seconds' in df.columns:
            df.plot(x='query', y='latency_seconds', kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Response Latency by Query')
            axes[1, 0].set_ylabel('Seconds')
            axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Source count
        if 'source_count' in df.columns:
            df.plot(x='query', y='source_count', kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Number of Sources Used by Query')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file)
        plt.close()
        
        print(f"Visualization saved to {output_file}")
        return output_file
    
    def identify_problem_queries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify queries that performed poorly
        
        Args:
            threshold: Minimum acceptable score (queries below this are flagged)
            
        Returns:
            List of problematic queries with their metrics
        """
        problem_queries = []
        
        for result in self.individual_results:
            metrics = result['metrics']
            
            # Define conditions for problematic queries
            is_problematic = (
                metrics.get('retrieval_precision', 1.0) < threshold or
                metrics.get('retrieval_recall', 1.0) < threshold or
                metrics.get('answer_relevance', 1.0) < threshold or
                metrics.get('rouge_l_f1', 1.0) < threshold
            )
            
            if is_problematic:
                problem_queries.append({
                    'query': result['query'],
                    'metrics': metrics,
                    'issues': self._identify_issues(metrics, threshold)
                })
        
        return problem_queries
    
    def _identify_issues(self, metrics: Dict[str, float], threshold: float) -> List[str]:
        """Identify specific issues with a query's metrics"""
        issues = []
        
        if metrics.get('retrieval_precision', 1.0) < threshold:
            issues.append("Low retrieval precision (too many irrelevant documents)")
        
        if metrics.get('retrieval_recall', 1.0) < threshold:
            issues.append("Low retrieval recall (missing relevant documents)")
        
        if metrics.get('answer_relevance', 1.0) < threshold:
            issues.append("Low answer relevance (answer not semantically similar to reference)")
        
        if metrics.get('rouge_l_f1', 1.0) < threshold:
            issues.append("Low ROUGE score (answer lacks textual overlap with reference)")
        
        if metrics.get('latency_seconds', 0.0) > 5.0:
            issues.append("High latency (response time too slow)")
        
        return issues


def analyze_rag_evaluation(results_file: str):
    """Analyze RAG evaluation results and generate report"""
    analyzer = RAGEvaluationAnalyzer(results_file=results_file)
    
    # Generate summary report
    report = analyzer.generate_summary_report()
    with open("rag_evaluation_report.md", "w") as f:
        f.write(report)
    
    # Create visualization
    analyzer.create_visualization()
    
    # Identify problem queries
    problem_queries = analyzer.identify_problem_queries(threshold=0.6)
    
    # Print summary to console
    print("\n=== RAG Evaluation Analysis Complete ===")
    print(f"Full report saved to: rag_evaluation_report.md")
    print(f"Charts saved to: rag_evaluation_charts.png")
    print(f"Identified {len(problem_queries)} problematic queries.")
    
    return {
        "report": report,
        "problem_queries": problem_queries
    }


if __name__ == "__main__":
    # Update with your results file
    results_file = "rag_evaluation_results_20250520_212602.json"
    
    # Run analysis
    analysis = analyze_rag_evaluation(results_file)