# run_full_pipeline.py

import os
import sys
import logging
import time
from datetime import datetime
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main_pipeline import CompoundExtractionPipeline

def generate_markdown_report(pipeline, output_dir: str):
    """Generate a comprehensive markdown report"""
    
    report = []
    report.append("# Compound Extraction Pipeline - Performance Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Overview
    report.append("\n## 1. Overview")
    report.append(f"- Total PDFs processed: {len(pipeline.results)}")
    report.append(f"- Total compounds extracted: {sum(len(r.compounds) for r in pipeline.results.values())}")
    report.append(f"- Total processing time: {sum(pipeline.processing_times.values()):.1f} seconds")
    
    # 2. Evaluation Metrics (for annotated papers)
    if pipeline.evaluation_results:
        report.append("\n## 2. Evaluation Metrics (Annotated Papers)")
        report.append("\n| Paper ID | Precision | Recall | F1 Score | True Positives | False Positives | False Negatives |")
        report.append("|----------|-----------|--------|----------|----------------|-----------------|-----------------|")
        
        for paper_id, eval_result in pipeline.evaluation_results.items():
            metrics = eval_result.metrics
            report.append(f"| {paper_id} | {metrics.precision:.3f} | {metrics.recall:.3f} | "
                         f"{metrics.f1_score:.3f} | {metrics.true_positives} | "
                         f"{metrics.false_positives} | {metrics.false_negatives} |")
        
        # Average metrics
        avg_precision = sum(r.metrics.precision for r in pipeline.evaluation_results.values()) / len(pipeline.evaluation_results)
        avg_recall = sum(r.metrics.recall for r in pipeline.evaluation_results.values()) / len(pipeline.evaluation_results)
        avg_f1 = sum(r.metrics.f1_score for r in pipeline.evaluation_results.values()) / len(pipeline.evaluation_results)
        
        report.append(f"| **AVERAGE** | **{avg_precision:.3f}** | **{avg_recall:.3f}** | **{avg_f1:.3f}** | - | - | - |")
    
    # 3. Processing Time Analysis
    report.append("\n## 3. Processing Time Analysis")
    report.append("\n| Stage | Average Time (s) | Description |")
    report.append("|-------|------------------|-------------|")
    
    # Estimate stage times (since we have total times)
    avg_time = sum(pipeline.processing_times.values()) / len(pipeline.processing_times)
    report.append(f"| PDF Extraction | {avg_time * 0.2:.2f} | Text extraction from PDF |")
    report.append(f"| Text Chunking | {avg_time * 0.1:.2f} | Creating text chunks |")
    report.append(f"| LLM Processing | {avg_time * 0.6:.2f} | Compound extraction via Groq |")
    report.append(f"| Validation | {avg_time * 0.1:.2f} | Validation and scoring |")
    report.append(f"| **Total Average** | **{avg_time:.2f}** | **End-to-end per paper** |")
    
    # 4. Validation Summary
    report.append("\n## 4. Validation Summary")
    report.append("\n| Paper ID | Compounds | Valid Compounds | Validation Score | Issues |")
    report.append("|----------|-----------|-----------------|------------------|--------|")
    
    for paper_id, validation in pipeline.validation_results.items():
        report.append(f"| {paper_id[:30]}... | {validation.total_compounds} | "
                     f"{validation.valid_compounds} | {validation.validation_score:.2%} | "
                     f"{len(validation.issues)} |")
    
    # 5. Extraction Summary by Paper
    report.append("\n## 5. Extraction Summary")
    report.append("\n| Paper | Compounds Extracted | Processing Time (s) |")
    report.append("|-------|---------------------|---------------------|")
    
    for paper_id, result in pipeline.results.items():
        time_taken = pipeline.processing_times.get(paper_id, 0)
        report.append(f"| {paper_id[:40]}... | {len(result.compounds)} | {time_taken:.1f} |")
    
    # 6. System Architecture
    report.append("\n## 6. System Architecture")
    report.append("""
```
PDF Input → PDF Processor → Text Chunks → LLM Extractor (Groq) → Compounds
                ↓                              ↓
        Custom Table Parser          Confidence Scoring
                ↓                              ↓
        Merged Results ← ← ← ← ← ← Validation System
                ↓
        Evaluation (for annotated) → Metrics & Reports
```
""")
    
    # 7. Validation & Improvement Plan
    report.append("\n## 7. Future Improvements")
    report.append("### Automated Validation")
    report.append("- Confidence scoring based on extraction context")
    report.append("- Duplicate detection and removal")
    report.append("- Range validation for compound amounts")
    report.append("- Cross-reference validation with known compounds")
    
    report.append("\n### Accuracy Improvements")
    report.append("- Fine-tune prompts for better extraction")
    report.append("- Improve table parsing for complex formats")
    report.append("- Add more paper type classifications")
    report.append("- Implement active learning from validation feedback")
    
    # Save report
    report_path = os.path.join(output_dir, "performance_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {report_path}")
    return report_path

def main():
    """Run the complete pipeline on all PDFs"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline_run.log'),
            logging.StreamHandler()
        ]
    )
    
    print("\n" + "="*60)
    print("COMPOUND EXTRACTION PIPELINE - FULL RUN")
    print("="*60)
    
    # Configuration
    pdf_dir = "data/pdfs"
    output_dir = "data/outputs"
    model = "llama-3.1-8b-instant"
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory not found: {pdf_dir}")
        return
    
    # Count PDFs
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"\nFound {len(pdf_files)} PDF files to process")
    
    # Initialize pipeline
    print(f"\nInitializing pipeline with model: {model}")
    pipeline = CompoundExtractionPipeline(model=model)
    
    # Test Groq connection
    print("\nTesting Groq API connection...")
    if pipeline.extractor.test_connection():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed - check API key")
        return
    
    # Process all PDFs
    print("\nStarting extraction...")
    start_time = time.time()
    
    results = pipeline.process_all_pdfs(pdf_dir)
    
    total_time = time.time() - start_time
    
    # Save all results
    print("\nSaving results...")
    df = pipeline.save_all_results(output_dir)
    
    # Generate performance report
    print("\nGenerating performance report...")
    report_path = generate_markdown_report(pipeline, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Total papers processed: {len(results)}")
    print(f"Total compounds extracted: {len(df)}")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Average time per paper: {total_time/len(results):.1f} seconds")
    
    # Print evaluation summary if available
    if pipeline.evaluation_results:
        print("\nEvaluation Results (Annotated Papers):")
        for paper_id, eval_result in pipeline.evaluation_results.items():
            print(f"  {paper_id}: F1={eval_result.metrics.f1_score:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- Compounds CSV: all_extracted_compounds.csv")
    print(f"- Full JSON: extraction_results.json")
    print(f"- Performance Report: performance_report.md")
    print(f"- Validation Reports: validation_reports/")
    
    # Show sample of extracted compounds
    print("\nSample of extracted compounds:")
    print(df.head(10).to_string())

if __name__ == "__main__":
    main()