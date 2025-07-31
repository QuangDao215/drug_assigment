# src/evaluator.py

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from difflib import SequenceMatcher
import pandas as pd
from datetime import datetime

from src.data_models import Compound, ExtractionResult

@dataclass
class CompoundMatch:
    """Represents a match between predicted and ground truth compound"""
    predicted: Compound
    ground_truth: Compound
    match_score: float
    match_type: str  # 'exact', 'fuzzy', 'partial'
    match_details: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics"""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    def calculate_metrics(self):
        """Calculate precision, recall, and F1 score"""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0
            
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0
            
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

@dataclass
class EvaluationResult:
    """Complete evaluation result for a paper"""
    paper_id: str
    metrics: EvaluationMetrics
    matches: List[CompoundMatch] = field(default_factory=list)
    false_positives: List[Compound] = field(default_factory=list)
    false_negatives: List[Compound] = field(default_factory=list)
    evaluation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'paper_id': self.paper_id,
            'metrics': {
                'precision': self.metrics.precision,
                'recall': self.metrics.recall,
                'f1_score': self.metrics.f1_score,
                'true_positives': self.metrics.true_positives,
                'false_positives': self.metrics.false_positives,
                'false_negatives': self.metrics.false_negatives
            },
            'matches': len(self.matches),
            'evaluation_time': self.evaluation_time
        }

class CompoundEvaluator:
    """Evaluates compound extraction results against ground truth"""
    
    def __init__(self, 
                 name_threshold: float = 0.85,
                 species_threshold: float = 0.90,
                 amount_tolerance: float = 0.05):
        """
        Initialize evaluator with matching thresholds
        
        Args:
            name_threshold: Minimum similarity for compound name matching (0-1)
            species_threshold: Minimum similarity for species name matching (0-1)
            amount_tolerance: Relative tolerance for amount matching (e.g., 0.05 = 5%)
        """
        self.logger = logging.getLogger(__name__)
        self.name_threshold = name_threshold
        self.species_threshold = species_threshold
        self.amount_tolerance = amount_tolerance
        
    def evaluate(self, 
                 predicted: ExtractionResult, 
                 ground_truth: ExtractionResult) -> EvaluationResult:
        """
        Evaluate predicted results against ground truth
        
        Args:
            predicted: Extracted compounds from model
            ground_truth: Annotated ground truth compounds
            
        Returns:
            EvaluationResult with metrics and match details
        """
        self.logger.info(f"Evaluating {predicted.paper_id}: "
                        f"{len(predicted.compounds)} predicted vs "
                        f"{len(ground_truth.compounds)} ground truth")
        
        # Initialize result
        result = EvaluationResult(
            paper_id=predicted.paper_id,
            metrics=EvaluationMetrics()
        )
        
        # Create copies for tracking
        unmatched_predicted = set(range(len(predicted.compounds)))
        unmatched_truth = set(range(len(ground_truth.compounds)))
        
        # Find matches
        for i, pred_compound in enumerate(predicted.compounds):
            best_match = None
            best_score = 0.0
            best_truth_idx = None
            
            for j in unmatched_truth:
                truth_compound = ground_truth.compounds[j]
                
                # Calculate match score
                match_score, match_details = self._calculate_match_score(
                    pred_compound, truth_compound
                )
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = CompoundMatch(
                        predicted=pred_compound,
                        ground_truth=truth_compound,
                        match_score=match_score,
                        match_type=self._get_match_type(match_score),
                        match_details=match_details
                    )
                    best_truth_idx = j
            
            # If we found a good match
            if best_match and best_score >= 0.7:  # Overall threshold
                result.matches.append(best_match)
                result.metrics.true_positives += 1
                unmatched_predicted.discard(i)
                unmatched_truth.discard(best_truth_idx)
                
                self.logger.debug(f"Matched: {pred_compound.name} -> "
                                f"{best_match.ground_truth.name} "
                                f"(score: {best_score:.2f})")
        
        # Handle unmatched compounds
        for i in unmatched_predicted:
            result.false_positives.append(predicted.compounds[i])
            result.metrics.false_positives += 1
            self.logger.debug(f"False positive: {predicted.compounds[i].name}")
            
        for j in unmatched_truth:
            result.false_negatives.append(ground_truth.compounds[j])
            result.metrics.false_negatives += 1
            self.logger.debug(f"False negative: {ground_truth.compounds[j].name}")
        
        # Calculate final metrics
        result.metrics.calculate_metrics()
        
        self.logger.info(f"Evaluation complete: "
                        f"Precision={result.metrics.precision:.2f}, "
                        f"Recall={result.metrics.recall:.2f}, "
                        f"F1={result.metrics.f1_score:.2f}")
        
        return result
    
    def _calculate_match_score(self, 
                              pred: Compound, 
                              truth: Compound) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall match score between two compounds
        
        Returns:
            Tuple of (overall_score, detail_scores)
        """
        details = {}
        
        # 1. Name similarity
        name_score = self._string_similarity(
            pred.name.lower(), 
            truth.name.lower()
        )
        details['name_score'] = name_score
        
        # 2. Species similarity (using normalized names)
        species_score = self._string_similarity(
            pred.species.normalized_name.lower(),
            truth.species.normalized_name.lower()
        )
        details['species_score'] = species_score
        
        # 3. Organism similarity
        organism_score = self._string_similarity(
            pred.organism.normalized_name.lower(),
            truth.organism.normalized_name.lower()
        )
        details['organism_score'] = organism_score
        
        # 4. Amount similarity
        amount_score = self._amount_similarity(pred.amount, truth.amount)
        details['amount_score'] = amount_score
        
        # Calculate weighted overall score
        # Name and amount are most important
        weights = {
            'name': 0.35,
            'species': 0.25,
            'organism': 0.15,
            'amount': 0.25
        }
        
        overall_score = (
            weights['name'] * name_score +
            weights['species'] * species_score +
            weights['organism'] * organism_score +
            weights['amount'] * amount_score
        )
        
        return overall_score, details
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        # Handle exact matches first
        if str1 == str2:
            return 1.0
            
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _amount_similarity(self, amount1, amount2) -> float:
        """
        Calculate similarity between two amounts
        Handles unit conversion and tolerance
        """
        # If units are different, try to convert
        if amount1.unit_type != amount2.unit_type:
            # If one is mass and can be converted
            if amount1.unit_type == 'mass' and amount2.unit_type == 'mass':
                val1_mg = amount1.to_mg()
                val2_mg = amount2.to_mg()
                if val1_mg and val2_mg:
                    return self._numeric_similarity(val1_mg, val2_mg)
            return 0.0  # Can't compare different unit types
        
        # Same unit type - compare values
        return self._numeric_similarity(amount1.value, amount2.value)
    
    def _numeric_similarity(self, val1: float, val2: float) -> float:
        """Calculate similarity between two numeric values"""
        if val1 == val2:
            return 1.0
            
        # Calculate relative difference
        max_val = max(abs(val1), abs(val2))
        if max_val == 0:
            return 1.0
            
        rel_diff = abs(val1 - val2) / max_val
        
        # Convert to similarity score (1 - relative difference)
        # But cap it at the tolerance level
        if rel_diff <= self.amount_tolerance:
            return 1.0
        else:
            return max(0, 1 - rel_diff)
    
    def _get_match_type(self, score: float) -> str:
        """Categorize match based on score"""
        if score >= 0.95:
            return 'exact'
        elif score >= 0.8:
            return 'fuzzy'
        else:
            return 'partial'
    
    def create_evaluation_report(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """
        Create a summary report from multiple evaluation results
        
        Args:
            results: List of evaluation results
            
        Returns:
            DataFrame with evaluation summary
        """
        summary_data = []
        
        for result in results:
            summary_data.append({
                'paper_id': result.paper_id,
                'precision': f"{result.metrics.precision:.3f}",
                'recall': f"{result.metrics.recall:.3f}",
                'f1_score': f"{result.metrics.f1_score:.3f}",
                'true_positives': result.metrics.true_positives,
                'false_positives': result.metrics.false_positives,
                'false_negatives': result.metrics.false_negatives,
                'total_matches': len(result.matches)
            })
        
        df = pd.DataFrame(summary_data)
        
        # Add average row
        avg_data = {
            'paper_id': 'AVERAGE',
            'precision': f"{df['precision'].astype(float).mean():.3f}",
            'recall': f"{df['recall'].astype(float).mean():.3f}",
            'f1_score': f"{df['f1_score'].astype(float).mean():.3f}",
            'true_positives': df['true_positives'].sum(),
            'false_positives': df['false_positives'].sum(), 
            'false_negatives': df['false_negatives'].sum(),
            'total_matches': df['total_matches'].sum()
        }
        
        df = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)
        
        return df
    
    def print_detailed_comparison(self, result: EvaluationResult, top_n: int = 10):
        """Print detailed comparison of matches and mismatches"""
        
        print(f"\n{'='*80}")
        print(f"Detailed Evaluation for {result.paper_id}")
        print(f"{'='*80}")
        
        print(f"\nMetrics:")
        print(f"  Precision: {result.metrics.precision:.3f}")
        print(f"  Recall: {result.metrics.recall:.3f}")
        print(f"  F1 Score: {result.metrics.f1_score:.3f}")
        
        if result.matches:
            print(f"\nTop {min(top_n, len(result.matches))} Matches:")
            for i, match in enumerate(sorted(result.matches, 
                                           key=lambda x: x.match_score, 
                                           reverse=True)[:top_n]):
                print(f"\n  {i+1}. Score: {match.match_score:.3f} ({match.match_type})")
                print(f"     Predicted: {match.predicted.name} - "
                      f"{match.predicted.amount.value} {match.predicted.amount.unit}")
                print(f"     Truth: {match.ground_truth.name} - "
                      f"{match.ground_truth.amount.value} {match.ground_truth.amount.unit}")
                print(f"     Details: {match.match_details}")
        
        if result.false_positives:
            print(f"\nFalse Positives (first {min(5, len(result.false_positives))}):")
            for i, fp in enumerate(result.false_positives[:5]):
                print(f"  {i+1}. {fp.name} ({fp.species.scientific_name}) - "
                      f"{fp.amount.value} {fp.amount.unit}")
        
        if result.false_negatives:
            print(f"\nFalse Negatives (first {min(5, len(result.false_negatives))}):")
            for i, fn in enumerate(result.false_negatives[:5]):
                print(f"  {i+1}. {fn.name} ({fn.species.scientific_name}) - "
                      f"{fn.amount.value} {fn.amount.unit}")