# test_pipeline_fixed.py - Test on annotated papers only

import sys
sys.path.append('..')

import logging
import os
from src.main_pipeline import CompoundExtractionPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize pipeline
pipeline = CompoundExtractionPipeline(model="llama-3.1-8b-instant")

# Define the actual PDF filenames
paper1_pdf = "coll-et-al-2011-neo-clerodane-diterpenoids-from-ajuga-bracteosa.pdf"
paper2_pdf = "J Sci Food Agric - 1999 - Zlatanov - Lipid composition of Bulgarian chokeberry  black currant and rose hip seed oils.pdf"

print("Testing on annotated papers...")

# Process paper1 (Ajuga bracteosa)
print(f"\nProcessing Paper 1: {paper1_pdf}")
paper1_result, paper1_validation = pipeline.process_single_pdf(
    f"data/pdfs/{paper1_pdf}",
    "paper1"
)

print(f"\nPaper 1 Results:")
print(f"Extracted: {len(paper1_result.compounds)} compounds")
print(f"Valid compounds: {paper1_validation.valid_compounds}")
print(f"Validation score: {paper1_validation.validation_score:.2%}")
print(f"Expected (ground truth): 17 compounds")

if paper1_result.compounds:
    print(f"\nFirst 5 extracted compounds:")
    for i, comp in enumerate(paper1_result.compounds[:5]):
        print(f"  {i+1}. {comp}")

# Get ground truth for comparison
gt_paper1 = pipeline.gt_extractor.get_paper_results("paper1")
if gt_paper1:
    print(f"\nGround truth first 5:")
    for i, comp in enumerate(gt_paper1.compounds[:5]):
        print(f"  {i+1}. {comp}")

# Process paper2 (Aronia, Ribes, Rosa)
print(f"\n{'='*60}")
print(f"\nProcessing Paper 2: {paper2_pdf[:50]}...")
paper2_result, paper2_validation = pipeline.process_single_pdf(
    f"data/pdfs/{paper2_pdf}",
    "paper2"
)

print(f"\nPaper 2 Results:")
print(f"Extracted: {len(paper2_result.compounds)} compounds")
print(f"Valid compounds: {paper2_validation.valid_compounds}")
print(f"Validation score: {paper2_validation.validation_score:.2%}")
print(f"Expected (ground truth): 30 compounds")

if paper2_result.compounds:
    print(f"\nFirst 5 extracted compounds:")
    for i, comp in enumerate(paper2_result.compounds[:5]):
        print(f"  {i+1}. {comp}")

# Display evaluation results
print(f"\n{'='*60}")
print("EVALUATION RESULTS")
print(f"{'='*60}")

if pipeline.evaluation_results:
    # Create evaluation summary
    summary_df = pipeline.evaluator.create_evaluation_report(list(pipeline.evaluation_results.values()))
    print("\n" + str(summary_df))
    
    # Show detailed comparison for paper1
    if 'paper1' in pipeline.evaluation_results:
        print(f"\n{'='*40}")
        print("DETAILED COMPARISON - PAPER 1")
        print(f"{'='*40}")
        pipeline.evaluator.print_detailed_comparison(pipeline.evaluation_results['paper1'], top_n=5)
    
    # Show detailed comparison for paper2
    if 'paper2' in pipeline.evaluation_results:
        print(f"\n{'='*40}")
        print("DETAILED COMPARISON - PAPER 2")
        print(f"{'='*40}")
        pipeline.evaluator.print_detailed_comparison(pipeline.evaluation_results['paper2'], top_n=5)

# Display validation issues summary
print(f"\n{'='*60}")
print("VALIDATION SUMMARY")
print(f"{'='*60}")

for paper_id, validation in [('paper1', paper1_validation), ('paper2', paper2_validation)]:
    print(f"\n{paper_id}:")
    print(f"  Total compounds: {validation.total_compounds}")
    print(f"  Valid compounds: {validation.valid_compounds}")
    print(f"  Validation score: {validation.validation_score:.2%}")
    
    if validation.issues:
        print(f"  Issues by type:")
        for issue_type, count in validation.issue_summary.items():
            print(f"    - {issue_type}: {count}")
    else:
        print("  No validation issues found!")

# Save results
print(f"\n{'='*60}")
print("Saving results...")

# Save intermediate results for both papers
pipeline.save_intermediate_results("paper1", paper1_result, paper1_validation)
pipeline.save_intermediate_results("paper2", paper2_result, paper2_validation)

# Generate evaluation summary
pipeline._generate_evaluation_summary()

print("\nTest complete! Check data/outputs/ for detailed results.")