# src/validator.py

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import pandas as pd

from src.data_models import Compound, ExtractionResult

@dataclass
class ValidationIssue:
    """Represents a validation issue found in a compound"""
    compound: Compound
    issue_type: str  # 'missing_field', 'invalid_format', 'suspicious_value', etc.
    severity: str    # 'error', 'warning', 'info'
    message: str
    field: Optional[str] = None
    
@dataclass 
class ValidationResult:
    """Results of validation for a set of compounds"""
    total_compounds: int = 0
    valid_compounds: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    duplicate_groups: List[List[Compound]] = field(default_factory=list)
    
    @property
    def validation_score(self) -> float:
        """Overall validation score (0-1)"""
        if self.total_compounds == 0:
            return 0.0
        return self.valid_compounds / self.total_compounds
    
    @property
    def issue_summary(self) -> Dict[str, int]:
        """Count issues by type"""
        summary = Counter()
        for issue in self.issues:
            summary[issue.issue_type] += 1
        return dict(summary)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to DataFrame for analysis"""
        if not self.issues:
            return pd.DataFrame()
        
        data = []
        for issue in self.issues:
            data.append({
                'compound_name': issue.compound.name,
                'issue_type': issue.issue_type,
                'severity': issue.severity,
                'message': issue.message,
                'field': issue.field
            })
        
        return pd.DataFrame(data)

class CompoundValidator:
    """Validates extracted compounds and assigns confidence scores"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Valid patterns
        self.species_pattern = re.compile(r'^[A-Z][a-z]+ [a-z]+')  # Genus species
        self.compound_name_pattern = re.compile(r'^[A-Za-z0-9\s\-,()αβγδ]+$')
        
        # Suspicious patterns
        self.suspicious_words = [
            'extract', 'fraction', 'standard', 'reference', 'blank',
            'solvent', 'mixture', 'unknown', 'compound'
        ]
        
        # Reasonable ranges for amounts (in mg)
        self.amount_ranges = {
            'isolation': (0.1, 5000),      # 0.1 mg to 5 g
            'analytical': (0.001, 10000)   # 1 μg to 10 g
        }
        
    def validate_extraction_result(self, 
                                 extraction_result: ExtractionResult,
                                 paper_type: str = 'isolation') -> ValidationResult:
        """
        Validate all compounds in an extraction result
        
        Args:
            extraction_result: The extraction result to validate
            paper_type: Type of paper ('isolation' or 'analytical')
            
        Returns:
            ValidationResult with issues and scores
        """
        self.logger.info(f"Validating {len(extraction_result.compounds)} compounds "
                        f"from {extraction_result.paper_id}")
        
        result = ValidationResult(total_compounds=len(extraction_result.compounds))
        
        # Check each compound
        for compound in extraction_result.compounds:
            issues = self._validate_compound(compound, paper_type)
            result.issues.extend(issues)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(compound, issues)
            result.confidence_scores[compound.name] = confidence
            
            # Count as valid if no error-level issues
            if not any(issue.severity == 'error' for issue in issues):
                result.valid_compounds += 1
        
        # Check for duplicates
        duplicates = self._find_duplicates(extraction_result.compounds)
        result.duplicate_groups = duplicates
        
        # Add duplicate issues
        for dup_group in duplicates:
            for compound in dup_group[1:]:  # Skip first occurrence
                result.issues.append(ValidationIssue(
                    compound=compound,
                    issue_type='duplicate',
                    severity='warning',
                    message=f"Duplicate of {dup_group[0].name}"
                ))
        
        # Check for suspicious patterns across all compounds
        pattern_issues = self._check_suspicious_patterns(extraction_result.compounds)
        result.issues.extend(pattern_issues)
        
        self.logger.info(f"Validation complete: {result.valid_compounds}/{result.total_compounds} valid, "
                        f"{len(result.issues)} issues found")
        
        return result
    
    def _validate_compound(self, compound: Compound, paper_type: str) -> List[ValidationIssue]:
        """Validate a single compound"""
        issues = []
        
        # 1. Check required fields
        if not compound.name or compound.name.strip() == "":
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='missing_field',
                severity='error',
                message='Compound name is missing',
                field='name'
            ))
        
        # 2. Validate compound name format
        if compound.name and not self.compound_name_pattern.match(compound.name):
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='invalid_format',
                severity='warning',
                message=f'Compound name contains invalid characters: {compound.name}',
                field='name'
            ))
        
        # 3. Check for suspicious compound names
        name_lower = compound.name.lower()
        for suspicious in self.suspicious_words:
            if suspicious in name_lower and not any(
                valid in name_lower for valid in ['extraction', 'fractional']
            ):
                issues.append(ValidationIssue(
                    compound=compound,
                    issue_type='suspicious_name',
                    severity='warning',
                    message=f'Compound name contains suspicious word: {suspicious}',
                    field='name'
                ))
                break
        
        # 4. Validate species name
        if not compound.species.scientific_name:
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='missing_field',
                severity='error',
                message='Species name is missing',
                field='species'
            ))
        elif not self.species_pattern.match(compound.species.scientific_name):
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='invalid_format',
                severity='warning',
                message=f'Species name format incorrect: {compound.species.scientific_name}',
                field='species'
            ))
        
        # 5. Validate organism/part
        if not compound.organism.name:
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='missing_field',
                severity='error',
                message='Organism part is missing',
                field='organism'
            ))
        
        # 6. Validate amount
        amount_issues = self._validate_amount(compound, paper_type)
        issues.extend(amount_issues)
        
        return issues
    
    def _validate_amount(self, compound: Compound, paper_type: str) -> List[ValidationIssue]:
        """Validate compound amount"""
        issues = []
        
        # Check if amount exists
        if compound.amount.value is None:
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='missing_field',
                severity='error',
                message='Amount value is missing',
                field='amount'
            ))
            return issues
        
        # Check if unit exists
        if not compound.amount.unit:
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='missing_field',
                severity='error',
                message='Amount unit is missing',
                field='amount'
            ))
            return issues
        
        # Convert to mg for range checking (if possible)
        if compound.amount.unit_type == 'mass':
            amount_mg = compound.amount.to_mg()
            if amount_mg:
                min_val, max_val = self.amount_ranges.get(paper_type, self.amount_ranges['isolation'])
                
                if amount_mg < min_val:
                    issues.append(ValidationIssue(
                        compound=compound,
                        issue_type='suspicious_value',
                        severity='warning',
                        message=f'Amount unusually small: {compound.amount.value} {compound.amount.unit} '
                               f'({amount_mg:.3f} mg)',
                        field='amount'
                    ))
                elif amount_mg > max_val:
                    issues.append(ValidationIssue(
                        compound=compound,
                        issue_type='suspicious_value',
                        severity='warning',
                        message=f'Amount unusually large: {compound.amount.value} {compound.amount.unit} '
                               f'({amount_mg:.3f} mg)',
                        field='amount'
                    ))
        
        # Check for zero or negative values
        if compound.amount.value <= 0:
            issues.append(ValidationIssue(
                compound=compound,
                issue_type='invalid_value',
                severity='error',
                message=f'Amount must be positive: {compound.amount.value}',
                field='amount'
            ))
        
        return issues
    
    def _find_duplicates(self, compounds: List[Compound]) -> List[List[Compound]]:
        """Find duplicate compounds"""
        duplicates = []
        seen = {}
        
        for compound in compounds:
            # Create a key for comparison
            key = (
                compound.name.lower().strip(),
                compound.species.normalized_name.lower(),
                compound.organism.normalized_name.lower()
            )
            
            if key in seen:
                # Found a duplicate
                found = False
                for dup_group in duplicates:
                    if seen[key] in dup_group:
                        dup_group.append(compound)
                        found = True
                        break
                
                if not found:
                    duplicates.append([seen[key], compound])
            else:
                seen[key] = compound
        
        return duplicates
    
    def _check_suspicious_patterns(self, compounds: List[Compound]) -> List[ValidationIssue]:
        """Check for suspicious patterns across all compounds"""
        issues = []
        
        # 1. Check for identical amounts (might indicate table extraction error)
        amount_counts = Counter()
        for compound in compounds:
            key = f"{compound.amount.value} {compound.amount.unit}"
            amount_counts[key] += 1
        
        # If more than 30% of compounds have the same amount, it's suspicious
        if compounds:
            most_common_amount, count = amount_counts.most_common(1)[0]
            if count > len(compounds) * 0.3 and count > 2:
                # Find compounds with this amount
                for compound in compounds:
                    if f"{compound.amount.value} {compound.amount.unit}" == most_common_amount:
                        issues.append(ValidationIssue(
                            compound=compound,
                            issue_type='pattern_issue',
                            severity='warning',
                            message=f'Suspicious: {count} compounds have identical amount ({most_common_amount})'
                        ))
        
        # 2. Check for sequential naming (compound 1, compound 2, etc.)
        numbered_compounds = []
        for compound in compounds:
            if re.search(r'compound\s*\d+', compound.name.lower()):
                numbered_compounds.append(compound)
        
        if len(numbered_compounds) > len(compounds) * 0.5:
            for compound in numbered_compounds:
                issues.append(ValidationIssue(
                    compound=compound,
                    issue_type='pattern_issue',
                    severity='info',
                    message='Generic numbered compound name - verify extraction'
                ))
        
        return issues
    
    def _calculate_confidence_score(self, 
                                  compound: Compound, 
                                  issues: List[ValidationIssue]) -> float:
        """
        Calculate confidence score for a compound (0-1)
        
        Factors:
        - Base confidence from extraction
        - Deductions for validation issues
        - Bonus for complete data
        """
        # Start with base confidence
        confidence = getattr(compound, 'confidence_score', 0.8)
        
        # Deduct for issues
        for issue in issues:
            if issue.severity == 'error':
                confidence -= 0.3
            elif issue.severity == 'warning':
                confidence -= 0.1
            elif issue.severity == 'info':
                confidence -= 0.05
        
        # Bonus for complete, well-formatted data
        if not issues:
            confidence += 0.1
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report"""
        report = []
        report.append(f"\nValidation Report")
        report.append("=" * 50)
        report.append(f"Total Compounds: {result.total_compounds}")
        report.append(f"Valid Compounds: {result.valid_compounds}")
        report.append(f"Validation Score: {result.validation_score:.2%}")
        
        if result.issues:
            report.append(f"\nIssues Found: {len(result.issues)}")
            report.append("\nIssue Summary:")
            for issue_type, count in result.issue_summary.items():
                report.append(f"  - {issue_type}: {count}")
            
            # Show top issues
            report.append("\nTop Issues:")
            for i, issue in enumerate(result.issues[:10]):
                report.append(f"\n{i+1}. {issue.severity.upper()}: {issue.message}")
                report.append(f"   Compound: {issue.compound.name}")
                if issue.field:
                    report.append(f"   Field: {issue.field}")
        
        if result.duplicate_groups:
            report.append(f"\nDuplicate Groups Found: {len(result.duplicate_groups)}")
            for i, group in enumerate(result.duplicate_groups[:5]):
                report.append(f"\n  Group {i+1}:")
                for compound in group:
                    report.append(f"    - {compound.name} ({compound.amount.value} {compound.amount.unit})")
        
        # Confidence score distribution
        if result.confidence_scores:
            scores = list(result.confidence_scores.values())
            avg_confidence = sum(scores) / len(scores)
            report.append(f"\nAverage Confidence Score: {avg_confidence:.2f}")
            report.append(f"  High confidence (>0.8): {sum(1 for s in scores if s > 0.8)}")
            report.append(f"  Medium confidence (0.5-0.8): {sum(1 for s in scores if 0.5 <= s <= 0.8)}")
            report.append(f"  Low confidence (<0.5): {sum(1 for s in scores if s < 0.5)}")
        
        return "\n".join(report)