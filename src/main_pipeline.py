# src/main_pipeline.py

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from tqdm import tqdm

from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker
from src.groq_extractor import GroqExtractor
from src.ground_truth_extractor import GroundTruthExtractor
from src.data_models import ExtractionResult
from src.evaluator import CompoundEvaluator, EvaluationResult
from src.validator import CompoundValidator, ValidationResult

class CompoundExtractionPipeline:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """Initialize all pipeline components"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker(chunk_size=2000, overlap=200)
        self.extractor = GroqExtractor(model=model)
        self.gt_extractor = GroundTruthExtractor()
        
        # NEW: Initialize evaluator and validator
        self.evaluator = CompoundEvaluator()
        self.validator = CompoundValidator()
        
        # Results storage
        self.results = {}
        self.processing_times = {}
        self.validation_results = {}  # NEW
        self.evaluation_results = {}  # NEW
        
        # Track annotated papers
        self.annotated_papers = ['paper1', 'paper2']
        
    def process_single_pdf(self, pdf_path: str, paper_id: str) -> Tuple[ExtractionResult, ValidationResult]:
        """
        Process a single PDF and extract compounds
        
        Returns:
            Tuple of (ExtractionResult, ValidationResult)
        """
        
        start_time = datetime.now()
        self.logger.info(f"Processing {paper_id}: {pdf_path}")
        
        try:
            # Step 1: Extract text from PDF
            self.logger.info("Extracting text from PDF...")
            text = self.pdf_processor.extract_text(pdf_path)
            
            # Step 1.5: Classify paper type
            from src.paper_classifier import PaperTypeClassifier
            classifier = PaperTypeClassifier()
            paper_type, confidence = classifier.classify_paper(text)
            self.logger.info(f"Detected paper type: {paper_type} (confidence: {confidence:.2f})")
            
            # Step 1.6: Check if custom table parsing is needed
            needs_custom_parsing = "[CUSTOM_TABLE_PARSING_NEEDED]" in text
            if needs_custom_parsing:
                self.logger.info("Custom table parsing detected as necessary")
                # Remove the marker from text
                text = text.replace("[CUSTOM_TABLE_PARSING_NEEDED]\n\n", "")
                
                # Extract compounds from text tables
                text_table_compounds = self.pdf_processor.extract_compounds_from_text_tables(
                    text, 
                    'analytical' if paper_type == 'analytical_chemistry' else 'isolation'
                )
                self.logger.info(f"Extracted {len(text_table_compounds)} compounds from text tables")
            else:
                text_table_compounds = []
            
            # Step 2: Create chunks with paper type awareness
            self.logger.info("Creating text chunks...")
            self.chunker.paper_type = paper_type
            chunks = self.chunker.create_smart_chunks(text)
            
            # Step 3: Extract compounds with paper type awareness
            self.logger.info("Extracting compounds...")
            self.extractor.paper_type = paper_type
            
            # Make sure extractor has prompt templates
            if not hasattr(self.extractor, 'prompt_templates'):
                from src.prompt_templates import PromptTemplates
                self.extractor.prompt_templates = PromptTemplates()
            
            result = self.extractor.extract_from_chunks(chunks)
            
            # Step 3.5: Add compounds from text tables if any
            if text_table_compounds:
                self.logger.info("Adding compounds from custom table parsing")
                from src.data_models import Compound, Species, Organism, Amount
                
                for comp_data in text_table_compounds:
                    try:
                        compound = Compound(
                            name=comp_data['compound_name'],
                            species=Species(scientific_name=comp_data['species']),
                            organism=Organism(name=comp_data['organism']),
                            amount=Amount(
                                value=comp_data['amount_value'],
                                unit=comp_data['amount_unit']
                            ),
                            confidence_score=0.85  # Good confidence for direct parsing
                        )
                        result.add_compound(compound)
                    except Exception as e:
                        self.logger.error(f"Error adding text table compound: {e}")
            
            result.paper_id = paper_id
            
            # Step 4: NEW - Validate extracted compounds
            self.logger.info("Validating extracted compounds...")
            validation_result = self.validator.validate_extraction_result(
                result, 
                paper_type='analytical' if paper_type == 'analytical_chemistry' else 'isolation'
            )
            
            # Update confidence scores in compounds
            for compound in result.compounds:
                if compound.name in validation_result.confidence_scores:
                    compound.confidence_score = validation_result.confidence_scores[compound.name]
            
            # Record processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_times[paper_id] = processing_time
            
            self.logger.info(f"Completed {paper_id} in {processing_time:.2f} seconds")
            self.logger.info(f"Extracted {len(result.compounds)} compounds "
                           f"({validation_result.valid_compounds} valid)")
            
            # Step 5: NEW - If this is an annotated paper, evaluate it
            if paper_id in self.annotated_papers:
                self.logger.info(f"Evaluating annotated paper {paper_id}...")
                evaluation_result = self._evaluate_annotated_paper(paper_id, result)
                if evaluation_result:
                    self.evaluation_results[paper_id] = evaluation_result
            
            return result, validation_result
            
        except Exception as e:
            self.logger.error(f"Error processing {paper_id}: {str(e)}")
            empty_result = ExtractionResult(paper_id=paper_id)
            empty_validation = ValidationResult()
            return empty_result, empty_validation
    
    def _evaluate_annotated_paper(self, paper_id: str, extraction_result: ExtractionResult) -> Optional[EvaluationResult]:
        """Evaluate extraction results against ground truth for annotated papers"""
        
        # Get ground truth
        ground_truth = self.gt_extractor.get_paper_results(paper_id)
        
        if not ground_truth:
            self.logger.warning(f"No ground truth found for {paper_id}")
            return None
        
        # Evaluate
        evaluation_result = self.evaluator.evaluate(extraction_result, ground_truth)
        
        # Log summary
        self.logger.info(f"Evaluation for {paper_id}:")
        self.logger.info(f"  Precision: {evaluation_result.metrics.precision:.3f}")
        self.logger.info(f"  Recall: {evaluation_result.metrics.recall:.3f}")
        self.logger.info(f"  F1 Score: {evaluation_result.metrics.f1_score:.3f}")
        
        return evaluation_result
    
    def process_all_pdfs(self, pdf_directory: str) -> Dict[str, ExtractionResult]:
        """Process all PDFs in a directory"""
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {}
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            paper_id = os.path.splitext(pdf_file)[0]
            
            extraction_result, validation_result = self.process_single_pdf(pdf_path, paper_id)
            results[paper_id] = extraction_result
            self.validation_results[paper_id] = validation_result
            
            # Save intermediate results
            self.save_intermediate_results(paper_id, extraction_result, validation_result)
        
        self.results = results
        
        # Generate evaluation summary if we processed annotated papers
        self._generate_evaluation_summary()
        
        return results
    
    def save_intermediate_results(self, paper_id: str, result: ExtractionResult, validation: ValidationResult):
        """Save results after each PDF to avoid data loss"""
        
        output_dir = "data/outputs/intermediate"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON with validation info
        output_path = os.path.join(output_dir, f"{paper_id}_extracted.json")
        
        data = {
            "paper_id": paper_id,
            "compound_count": len(result.compounds),
            "valid_compound_count": validation.valid_compounds,
            "validation_score": validation.validation_score,
            "compounds": [comp.to_dict() for comp in result.compounds],
            "validation_issues": len(validation.issues),
            "issue_summary": validation.issue_summary,
            "processing_time": self.processing_times.get(paper_id, 0)
        }
        
        # Add evaluation metrics if available
        if paper_id in self.evaluation_results:
            eval_result = self.evaluation_results[paper_id]
            data["evaluation"] = {
                "precision": eval_result.metrics.precision,
                "recall": eval_result.metrics.recall,
                "f1_score": eval_result.metrics.f1_score
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_evaluation_summary(self):
        """Generate summary of evaluation results for annotated papers"""
        
        if not self.evaluation_results:
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATION SUMMARY FOR ANNOTATED PAPERS")
        self.logger.info("="*60)
        
        # Create evaluation report
        eval_results = list(self.evaluation_results.values())
        summary_df = self.evaluator.create_evaluation_report(eval_results)
        
        print("\n" + str(summary_df))
        
        # Save evaluation report
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False)
    
    def save_all_results(self, output_dir: str = "data/outputs"):
        """Save all extraction results with validation and evaluation data"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all results into one dataframe
        all_compounds = []
        
        for paper_id, result in self.results.items():
            for compound in result.compounds:
                comp_dict = compound.to_dict()
                comp_dict['paper_id'] = paper_id
                
                # Add validation info
                if paper_id in self.validation_results:
                    validation = self.validation_results[paper_id]
                    comp_dict['validation_score'] = validation.validation_score
                    comp_dict['validation_issues'] = len(validation.issues)
                
                all_compounds.append(comp_dict)
        
        # Save as CSV
        df = pd.DataFrame(all_compounds)
        df.to_csv(os.path.join(output_dir, "all_extracted_compounds.csv"), index=False)
        
        # Create comprehensive JSON output
        output_data = {
            "extraction_date": datetime.now().isoformat(),
            "total_papers": len(self.results),
            "total_compounds": len(all_compounds),
            "processing_times": self.processing_times,
            "papers": {}
        }
        
        # Add detailed info for each paper
        for paper_id, result in self.results.items():
            paper_data = {
                "compound_count": len(result.compounds),
                "compounds": [comp.to_dict() for comp in result.compounds],
                "processing_time": self.processing_times.get(paper_id, 0)
            }
            
            # Add validation results
            if paper_id in self.validation_results:
                validation = self.validation_results[paper_id]
                paper_data["validation"] = {
                    "valid_compounds": validation.valid_compounds,
                    "validation_score": validation.validation_score,
                    "total_issues": len(validation.issues),
                    "issue_summary": validation.issue_summary
                }
            
            # Add evaluation results for annotated papers
            if paper_id in self.evaluation_results:
                eval_result = self.evaluation_results[paper_id]
                paper_data["evaluation"] = {
                    "precision": eval_result.metrics.precision,
                    "recall": eval_result.metrics.recall,
                    "f1_score": eval_result.metrics.f1_score,
                    "true_positives": eval_result.metrics.true_positives,
                    "false_positives": eval_result.metrics.false_positives,
                    "false_negatives": eval_result.metrics.false_negatives
                }
            
            output_data["papers"][paper_id] = paper_data
        
        # Calculate overall metrics
        if self.evaluation_results:
            avg_precision = sum(r.metrics.precision for r in self.evaluation_results.values()) / len(self.evaluation_results)
            avg_recall = sum(r.metrics.recall for r in self.evaluation_results.values()) / len(self.evaluation_results)
            avg_f1 = sum(r.metrics.f1_score for r in self.evaluation_results.values()) / len(self.evaluation_results)
            
            output_data["overall_evaluation"] = {
                "annotated_papers_evaluated": len(self.evaluation_results),
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "average_f1_score": avg_f1
            }
        
        # Save comprehensive JSON
        with open(os.path.join(output_dir, "extraction_results.json"), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Generate validation reports
        self._save_validation_reports(output_dir)
        
        self.logger.info(f"Saved all results to {output_dir}")
        
        return df
    
    def _save_validation_reports(self, output_dir: str):
        """Save detailed validation reports"""
        
        validation_dir = os.path.join(output_dir, "validation_reports")
        os.makedirs(validation_dir, exist_ok=True)
        
        # Combine all validation issues
        all_issues = []
        
        for paper_id, validation in self.validation_results.items():
            # Save individual validation report
            report_path = os.path.join(validation_dir, f"{paper_id}_validation.txt")
            with open(report_path, 'w') as f:
                f.write(self.validator.generate_validation_report(validation))
            
            # Collect issues for summary
            if validation.issues:
                issues_df = validation.to_dataframe()
                issues_df['paper_id'] = paper_id
                all_issues.append(issues_df)
        
        # Save combined issues CSV
        if all_issues:
            combined_issues = pd.concat(all_issues, ignore_index=True)
            combined_issues.to_csv(os.path.join(output_dir, "all_validation_issues.csv"), index=False)
            
            # Create issue summary
            issue_summary = combined_issues.groupby(['issue_type', 'severity']).size().reset_index(name='count')
            issue_summary.to_csv(os.path.join(output_dir, "validation_issue_summary.csv"), index=False)
    
    def evaluate_specific_papers(self, paper_ids: List[str]):
        """
        Evaluate specific papers against ground truth
        Useful for testing on annotated papers only
        """
        
        self.logger.info(f"Evaluating {len(paper_ids)} specific papers")
        
        for paper_id in paper_ids:
            if paper_id not in self.results:
                self.logger.warning(f"No extraction results found for {paper_id}")
                continue
            
            if paper_id not in self.annotated_papers:
                self.logger.warning(f"{paper_id} is not an annotated paper")
                continue
            
            extraction_result = self.results[paper_id]
            eval_result = self._evaluate_annotated_paper(paper_id, extraction_result)
            
            if eval_result:
                self.evaluation_results[paper_id] = eval_result
                # Print detailed comparison
                self.evaluator.print_detailed_comparison(eval_result, top_n=10)
        
        # Generate summary
        self._generate_evaluation_summary()