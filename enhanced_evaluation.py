# analyze_evaluation.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Added for finding latest results file
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
            try:
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded evaluation results from {results_file}")
            except FileNotFoundError:
                print(f"Error: Results file not found at {results_file}")
                raise
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {results_file}")
                raise
        elif results_data:
            self.results = results_data
            print("Initialized evaluation results from provided data.")
        else:
            raise ValueError("Either results_file or results_data must be provided")

        # Ensure necessary keys exist
        if 'aggregate_metrics' not in self.results or 'individual_results' not in self.results:
             raise ValueError("Results data must contain 'aggregate_metrics' and 'individual_results' keys.")

        self.aggregate_metrics = self.results['aggregate_metrics']
        self.individual_results = self.results['individual_results']

        # Configure plotting style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.dpi'] = 100 # Increase resolution

    def generate_summary_report(self) -> str:
        """Generate a text summary report of evaluation results"""
        report = []

        # Header
        report.append("# RAG Evaluation Summary Report")
        report.append(f"\nEvaluation run on {len(self.individual_results)} queries.\n")
        report.append("## Overall Performance Metrics\n")

        # Format aggregate metrics
        metrics_table = "| Metric | Value |\n|--------|-------|\n"
        # Sort metrics alphabetically for consistent reporting
        sorted_metrics = sorted(self.aggregate_metrics.items())
        for metric, value in sorted_metrics:
            # Format metrics nicely
            if value is None:
                formatted_value = "N/A"
            elif metric == 'latency_seconds':
                formatted_value = f"{value:.2f}s"
            elif metric == 'source_count':
                formatted_value = f"{value:.1f}"
            # Check for potential percentage metrics (scores typically range 0-1 and are floats)
            elif isinstance(value, float) and 0.0 <= value <= 1.0:
                 formatted_value = f"{value:.2%}"
            else:
                formatted_value = str(value) # Handle other potential data types

            metrics_table += f"| {metric.replace('_', ' ').title()} | {formatted_value} |\n"

        report.append(metrics_table)

        # Interpretation
        report.append("\n## Interpretation\n")

        # Analyze retrieval quality (only if metrics are present)
        retrieval_precision = self.aggregate_metrics.get('retrieval_precision', None)
        retrieval_recall = self.aggregate_metrics.get('retrieval_recall', None)

        if retrieval_precision is not None and retrieval_recall is not None:
            report.append("### Retrieval Quality\n")
            if retrieval_precision > 0.8 and retrieval_recall > 0.8:
                retrieval_assessment = "Excellent retrieval performance. The system is finding most relevant documents with high precision."
            elif retrieval_precision > 0.6 and retrieval_recall > 0.6:
                retrieval_assessment = "Good retrieval performance. The system retrieves most relevant documents but with some irrelevant results."
            elif retrieval_precision > retrieval_recall:
                retrieval_assessment = "The system is precise but may be missing some relevant documents (low recall). Consider adjusting retrieval to be more inclusive."
            elif retrieval_recall > retrieval_precision:
                retrieval_assessment = "The system finds many relevant documents but also includes too many irrelevant ones (low precision). Consider making retrieval more selective."
            else:
                retrieval_assessment = "Retrieval performance needs improvement. Consider retraining or adjusting the vector database."
            report.append(retrieval_assessment)
        else:
            report.append("### Retrieval Quality\nRetrieval metrics were not available in the results (relevant_docs not provided in test data).")


        # Analyze answer quality (only if metrics are present)
        answer_relevance = self.aggregate_metrics.get('answer_relevance', None)
        rouge_score = self.aggregate_metrics.get('rouge_l_f1', None)
        # answer_correctness = self.aggregate_metrics.get('answer_correctness', None) # Currently a proxy or requires LLM eval

        if answer_relevance is not None or rouge_score is not None: # Check if at least one answer metric is available
            report.append("\n### Answer Quality\n")
            if answer_relevance is not None and rouge_score is not None:
                if answer_relevance > 0.85 and rouge_score > 0.6:
                     answer_assessment = "Excellent answer quality. Responses are highly relevant and closely match reference answers."
                elif answer_relevance > 0.7 and rouge_score > 0.4:
                    answer_assessment = "Good answer quality. Responses are relevant but may miss some details from reference answers."
                elif answer_relevance > 0.6 or rouge_score > 0.3: # Slightly lower threshold for 'acceptable'
                    answer_assessment = "Acceptable answer quality, but there is room for improvement in relevance or textual overlap."
                else:
                    answer_assessment = "Answer quality needs significant improvement. Responses may not be relevant or lack factual basis from the context."
                report.append(answer_assessment)
            elif answer_relevance is not None:
                 report.append(f"Answer relevance score ({answer_relevance:.2%}) suggests the semantic meaning is captured, but ROUGE score is not available.")
            elif rouge_score is not None:
                 report.append(f"ROUGE-L F1 score ({rouge_score:.2%}) suggests some textual overlap, but semantic relevance score is not available.")
        else:
             report.append("\n### Answer Quality\nAnswer metrics were not available in the results (ground truth answer not provided in test data).")

        # Analyze performance characteristics (only if metrics are present)
        latency = self.aggregate_metrics.get('latency_seconds', None)
        sources = self.aggregate_metrics.get('source_count', None)

        if latency is not None:
            report.append("\n### Performance Characteristics\n")
            if latency > 5:
                latency_assessment = f"System latency ({latency:.2f}s) is high. Consider optimizing the retrieval process or using a faster LLM."
            elif latency > 2: # Moderate latency
                 latency_assessment = f"System latency ({latency:.2f}s) is moderate. Acceptable for some applications, but could be improved."
            else:
                latency_assessment = f"System latency ({latency:.2f}s) is good."

            report.append(latency_assessment)

        if sources is not None:
             report.append(f"The system used an average of {sources:.1f} sources per query.")


        # Recommendations
        report.append("\n## Recommendations\n")

        recommendations = []
        # Base recommendations on thresholds (adjust thresholds as needed)
        if retrieval_precision is not None and retrieval_precision < 0.7:
            recommendations.append("- **Improve retrieval precision:** The system retrieves too many irrelevant documents. Consider refining the document chunking strategy, using a re-ranking step, or improving the embedding model.")
        if retrieval_recall is not None and retrieval_recall < 0.7:
            recommendations.append("- **Improve retrieval recall:** The system is missing relevant documents. Consider adjusting the retrieval (e.g., increase `k`), improving the embedding model, or ensuring comprehensive documents are in the index.")
        if answer_relevance is not None and answer_relevance < 0.7:
             recommendations.append("- **Improve answer relevance:** The generated answers don't always capture the semantic meaning of the ground truth. Refine the prompt template to better guide the LLM based on the retrieved context.")
        if rouge_score is not None and rouge_score < 0.5:
            recommendations.append("- **Improve answer completeness/overlap:** The answers lack sufficient textual overlap with the ground truth. Ensure retrieved context is comprehensive and relevant, and refine the prompt to encourage detailed answers.")
        # Add correctness if you implement LLM evaluation for it
        # if answer_correctness is not None and answer_correctness < 0.7:
        #     recommendations.append("- **Improve answer correctness:** LLM evaluation suggests factual inaccuracies. Review retrieved documents for accuracy and refine LLM prompt instructions on factual grounding.")

        # Check latency threshold
        if latency is not None and latency > 3: # Recommendation trigger threshold
            recommendations.append("- **Reduce latency:** Response time is slow. Optimize the retrieval process (e.g., hardware, vector store tuning) or consider a faster LLM implementation or model.")

        # Check source count - maybe recommend fewer sources if precision is low?
        if sources is not None and retrieval_precision is not None and retrieval_precision < 0.6 and sources > 5:
             recommendations.append(f"- **Review source count:** The system uses many sources ({sources:.1f}) but precision is low ({retrieval_precision:.2%}). Consider reducing the number of retrieved documents (`k`) or implementing re-ranking to focus on the most relevant ones.")


        if not recommendations:
            report.append("- The system is performing well across the evaluated metrics. Consider expanding the test set with more complex and diverse queries to identify edge cases.")
        else:
             report.append("\n".join(recommendations))

        # Add a section about problematic queries
        report.append("\n## Problematic Queries (Below Threshold)")
        report.append("Details for queries performing below the set threshold are saved in `rag_problem_queries.json`.")


        return "\n".join(report)

    def create_visualization(self, output_file: str = "rag_evaluation_charts.png"):
        """Create visualization of evaluation metrics"""

        if not self.individual_results:
            print("No individual results available to create visualization.")
            return None

        # Convert individual results to DataFrame for easier plotting
        rows = []
        for result in self.individual_results:
            # Use more characters for query labels, handle very long queries
            truncated_query = result.get('query', 'N/A')
            truncated_query = truncated_query[:80] + '...' if len(truncated_query) > 80 else truncated_query # Use up to 80 chars

            row = {'query': truncated_query}
            # Add metrics, handle potential missing metrics if test data didn't provide ground truth (will be None in results)
            metrics = result.get('metrics', {})
            row['retrieval_precision'] = metrics.get('retrieval_precision', None)
            row['retrieval_recall'] = metrics.get('retrieval_recall', None)
            row['answer_relevance'] = metrics.get('answer_relevance', None)
            row['rouge_l_f1'] = metrics.get('rouge_l_f1', None)
            row['latency_seconds'] = metrics.get('latency_seconds', None)
            row['source_count'] = metrics.get('source_count', None)
            # Add other metrics if they are added to RAGEvalMetrics and calculated

            rows.append(row)

        df = pd.DataFrame(rows)
        # Drop columns that are entirely None or pd.NA (metrics not calculated for any query)
        df = df.dropna(axis=1, how='all')

        # Set query as index for plotting
        if 'query' in df.columns:
             df = df.set_index('query')
        else:
             print("DataFrame does not have a 'query' column. Cannot create plots.")
             return None


        if df.empty or df.shape[1] == 0: # Only index remains
             print("No plotable metrics data available in results.")
             return None

        # Identify which plots to create based on available columns
        plot_configs = []
        if 'retrieval_precision' in df.columns and 'retrieval_recall' in df.columns:
            plot_configs.append({'title': 'Retrieval Performance by Query', 'y_cols': ['retrieval_precision', 'retrieval_recall'], 'ylabel': 'Score (0-1)', 'ylim': (0, 1)})

        # Decide which answer metrics to plot together (plot only available ones)
        answer_cols = [col for col in ['answer_relevance', 'rouge_l_f1', 'answer_correctness'] if col in df.columns]
        if answer_cols:
             plot_configs.append({'title': 'Answer Quality by Query', 'y_cols': answer_cols, 'ylabel': 'Score (0-1)', 'ylim': (0, 1)})

        if 'latency_seconds' in df.columns:
            plot_configs.append({'title': 'Response Latency by Query', 'y_cols': ['latency_seconds'], 'ylabel': 'Seconds', 'ylim': (0, df['latency_seconds'].max() * 1.1 or 1)}) # Dynamic Y limit
        if 'source_count' in df.columns:
            plot_configs.append({'title': 'Number of Sources Used by Query', 'y_cols': ['source_count'], 'ylabel': 'Count', 'ylim': (0, df['source_count'].max() * 1.1 or 1)}) # Dynamic Y limit


        if not plot_configs:
            print("No plotable metrics found in the results.")
            return None

        # Calculate grid size
        n_cols = 2
        n_rows = (len(plot_configs) + n_cols - 1) // n_cols # Calculate needed rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5.5), squeeze=False) # Use squeeze=False for consistent 2D array
        axes = axes.flatten() # Flatten axes array for easy iteration

        # Create plots based on configurations
        for i, config in enumerate(plot_configs):
            ax = axes[i]
            y_cols = config['y_cols']
            plot_df = df[y_cols] # Select only columns for this plot

            plot_df.plot(kind='bar', ax=ax, ylim=config['ylim'])
            ax.set_title(config['title'])
            ax.set_ylabel(config['ylabel'])
            ax.tick_params(axis='x', rotation=45, ha='right') # Use tick_params for rotation
            ax.set_xlabel("Query (Truncated)") # Label X axis


            if len(y_cols) == 1:
                ax.get_legend().remove() # Remove legend for single-column plots

        # Hide any unused subplots
        for i in range(len(plot_configs), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to make space for suptitle
        fig.suptitle('RAG System Evaluation Results', fontsize=16) # Add overall title
        plt.savefig(output_file)
        plt.close()

        print(f"Visualization saved to {output_file}")
        return output_file

    def identify_problem_queries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify queries that performed poorly based on metrics falling below a threshold.

        Args:
            threshold: Minimum acceptable score (queries below this are flagged).
                       Applied to retrieval_precision, retrieval_recall,
                       answer_relevance, and rouge_l_f1 if available.

        Returns:
            List of problematic queries with their metrics and details.
        """
        problem_queries = []

        for result in self.individual_results:
            metrics = result.get('metrics', {}) # Get metrics safely
            query = result.get('query', 'N/A')

            # Define conditions for problematic queries based on available metrics
            is_problematic = False
            # Check metrics only if they exist (are not None) and are below threshold
            if metrics.get('retrieval_precision') is not None and metrics['retrieval_precision'] < threshold:
                 is_problematic = True
            if metrics.get('retrieval_recall') is not None and metrics['retrieval_recall'] < threshold:
                 is_problematic = True
            if metrics.get('answer_relevance') is not None and metrics['answer_relevance'] < threshold:
                 is_problematic = True
            if metrics.get('rouge_l_f1') is not None and metrics['rouge_l_f1'] < threshold:
                 is_problematic = True
            # Add checks for other relevant metrics like answer_correctness if available and used

            # Also flag queries with significant errors during processing
            if result.get('rag_answer', '').startswith('Error processing query:'):
                 is_problematic = True


            if is_problematic:
                problem_queries.append({
                    'query': query,
                    'metrics': metrics,
                    'issues': self._identify_issues(metrics, threshold, result.get('rag_answer', '')), # Pass rag answer to identify issues
                    'rag_answer': result.get('rag_answer', 'N/A'), # Include RAG's generated answer
                    'ground_truth_answer': result.get('ground_truth_answer', 'N/A'), # Include Ground Truth answer
                    'ground_truth_relevant_docs': result.get('ground_truth_relevant_docs', []), # Include Ground Truth relevant docs IDs
                    'retrieved_docs_info': result.get('retrieved_docs_info', []) # Include info about retrieved docs
                })

        return problem_queries

    def _identify_issues(self, metrics: Dict[str, float], threshold: float, rag_answer: str) -> List[str]:
        """Identify specific issues with a query's metrics based on threshold"""
        issues = []

        # Check for errors
        if rag_answer.startswith('Error processing query:'):
             issues.append(f"Processing Error: {rag_answer}")
             return issues # If error occurred, other metrics might be unreliable, just report the error

        # Check metrics against threshold (only if metric exists)
        if metrics.get('retrieval_precision') is not None and metrics['retrieval_precision'] < threshold:
            issues.append(f"Low retrieval precision ({metrics['retrieval_precision']:.2%}) < {threshold:.0%}) - too many irrelevant documents retrieved.")

        if metrics.get('retrieval_recall') is not None and metrics['retrieval_recall'] < threshold:
            issues.append(f"Low retrieval recall ({metrics['retrieval_recall']:.2%}) < {threshold:.0%}) - missing relevant documents.")

        if metrics.get('answer_relevance') is not None and metrics['answer_relevance'] < threshold:
            issues.append(f"Low answer relevance ({metrics['answer_relevance']:.2%}) < {threshold:.0%}) - answer not semantically similar to reference.")

        if metrics.get('rouge_l_f1') is not None and metrics['rouge_l_f1'] < threshold:
            issues.append(f"Low ROUGE-L F1 ({metrics['rouge_l_f1']:.2%}) < {threshold:.0%}) - answer lacks textual overlap with reference.")

        # Add checks for other metrics like correctness if applicable
        # if metrics.get('answer_correctness') is not None and metrics['answer_correctness'] < threshold:
        #      issues.append(f"Low answer correctness ({metrics['answer_correctness']:.2%}) < {threshold:.0%}) - answer may be factually incorrect.")


        # Specific threshold for latency issue reporting (independent of general threshold)
        latency_threshold_seconds = 5.0
        if metrics.get('latency_seconds', 0.0) > latency_threshold_seconds:
            issues.append(f"High latency ({metrics['latency_seconds']:.2f}s) > {latency_threshold_seconds:.1f}s).")

        # You could also add a check for source_count if that's relevant (e.g., too few sources?)

        return issues


def analyze_rag_evaluation(results_file: str):
    """Analyze RAG evaluation results and generate report and visualizations"""
    try:
        analyzer = RAGEvaluationAnalyzer(results_file=results_file)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Failed to initialize analyzer: {e}")
        return None # Exit if initialization fails

    # Generate summary report
    report = analyzer.generate_summary_report()
    report_filename = "rag_evaluation_report.md"
    with open(report_filename, "w") as f:
        f.write(report)
    print(f"Summary report saved to: {report_filename}")

    # Create visualization
    chart_filename = analyzer.create_visualization()


    # Identify problem queries
    # You can adjust the threshold here
    problem_threshold = 0.5 # Flag queries where any key metric is below 50%
    problem_queries = analyzer.identify_problem_queries(threshold=problem_threshold)
    problem_queries_filename = "rag_problem_queries.json"
    with open(problem_queries_filename, "w") as f:
        # Add indentation for readability
        json.dump(problem_queries, f, indent=2)
    print(f"Details of problematic queries (metrics < {problem_threshold:.0%}) saved to: {problem_queries_filename}")


    # Print summary to console
    print("\n=== RAG Evaluation Analysis Complete ===")
    print(f"Full report saved to: {report_filename}")
    if chart_filename:
        print(f"Charts saved to: {chart_filename}")
    print(f"Identified {len(problem_queries)} problematic queries (below threshold {problem_threshold:.0%}).")
    if problem_queries:
        print(f"Details of problematic queries saved to: {problem_queries_filename}")


    return {
        "report": report,
        "problem_queries": problem_queries,
        "chart_file": chart_filename
    }


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Update this path to the *actual* JSON file generated by your evaluation script
    # after you run it. Look for a file named rag_evaluation_results_YYYYMMDD_HHMMSS.json
    # in the same directory where you ran the evaluation script.
    # You can manually update this line, or use the automatic latest file finder below.

    # Example of finding the latest file:
    results_dir = "." # Directory where results are saved (usually the current directory)
    list_of_files = [f for f in os.listdir(results_dir) if f.startswith('rag_evaluation_results_') and f.endswith('.json')]
    if not list_of_files:
        print("No evaluation results files found matching 'rag_evaluation_results_*.json'")
        print("Please run the evaluation script ('python run_evaluation.py') first to generate results.")
        results_file_to_analyze = None
    else:
        # Sort files by modification time and get the latest
        latest_file_path = max([os.path.join(results_dir, f) for f in list_of_files], key=os.path.getmtime)
        results_file_to_analyze = latest_file_path
        print(f"Found latest results file: {results_file_to_analyze}")


    if results_file_to_analyze:
        # Run analysis
        print("\nStarting RAG evaluation analysis...")
        analysis = analyze_rag_evaluation(results_file_to_analyze)
    else:
        print("\nAnalysis aborted because no results file was found.")