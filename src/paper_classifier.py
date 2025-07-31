import re
from typing import Dict, Tuple

class PaperTypeClassifier:
    def __init__(self):
        self.type_indicators = {
            'isolation_chemistry': {
                'keywords': [
                    'isolated', 'yielded', 'afforded', 'obtained',
                    'fractionation', 'chromatography', 'crystallization',
                    'new compounds', 'novel compounds', 'compound 1',
                    'white powder', 'yellow crystals', 'colorless oil'
                ],
                'patterns': [
                    r'compound\s+\d+.*\(\d+\.?\d*\s*mg\)',
                    r'yielded.*\(\d+\.?\d*\s*mg\)',
                    r'afforded.*as.*(?:powder|crystals|oil)'
                ]
            },
            'analytical_chemistry': {
                'keywords': [
                    'composition', 'content', 'fatty acid', 'tocopherol',
                    'analyzed', 'determined', 'quantified', 'measured',
                    'g/kg oil', 'mg/kg', 'percentage', 'profile',
                    'gc-ms', 'hplc', 'spectrophotometric'
                ],
                'patterns': [
                    r'content.*was.*\d+\.?\d*\s*(?:g|mg)/kg',
                    r'composition.*(?:oil|extract)',
                    r'fatty acid.*profile',
                    r'Table\s+\d+.*composition'
                ]
            }
        }
    
    def classify_paper(self, full_text: str, first_chunks: list = None) -> Tuple[str, float]:
        """
        Classify paper type based on content
        Returns: (paper_type, confidence)
        """
        # Use first 5000 characters or first few chunks
        sample_text = full_text[:5000].lower()
        
        scores = {}
        for paper_type, indicators in self.type_indicators.items():
            score = 0
            
            # Check keywords
            for keyword in indicators['keywords']:
                score += sample_text.count(keyword) * 1
            
            # Check patterns (worth more)
            for pattern in indicators['patterns']:
                matches = re.findall(pattern, sample_text, re.IGNORECASE)
                score += len(matches) * 3
            
            scores[paper_type] = score
        
        # Determine type
        total_score = sum(scores.values())
        if total_score == 0:
            return 'unknown', 0.0
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / total_score
        
        return best_type, confidence