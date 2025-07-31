# src/custom_table_parser.py

import re
from typing import List, Dict, Tuple, Optional
import logging

class CustomTableParser:
    """Parse compound data from text-based tables in analytical chemistry papers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Compound name patterns
        self.compound_patterns = [
            # Fatty acids
            r'(?:Linoleic|Oleic|Palmitic|Stearic|Arachidic)\s*acid',
            # Tocopherols
            r'[αβγδ]?-?Tocopherol',
            r'(?:alpha|beta|gamma|delta)-Tocopherol',
            r'a-Tocopherol', r'b-Tocopherol', r'g-Tocopherol', r'd-Tocopherol',
            # Sterols
            r'(?:Cholesterol|Campesterol|Stigmasterol|Sitosterol|Avenasterol|Brassicasterol)',
            r'[βb]-Sitosterol',
            r'D\d+-(?:Stigmasterol|Avenasterol)',
            # Phospholipids
            r'Phosphatidyl(?:choline|ethanolamine|inositol)',
            r'(?:PC|PE|PI|PA)\b',
            # General pattern for compounds
            r'[A-Z][a-z]+(?:\s+[a-z]+)*\s*(?:acid|ester|ol|anol|sterol)'
        ]
        
        # Sample/species patterns
        self.species_patterns = {
            'chokeberry': ['Aronia melanocarpa', 'chokeberry', 'Aronia'],
            'blackcurrant': ['Ribes nigrum', 'black currant', 'blackcurrant'],
            'rosehip': ['Rosa canina', 'rose hip', 'rosehip']
        }
    
    def parse_analytical_table(self, text: str) -> List[Dict]:
        """
        Parse compound data from text-based analytical tables
        
        Args:
            text: PDF text containing table data
            
        Returns:
            List of compound dictionaries
        """
        compounds = []
        lines = text.split('\n')
        
        # Find compound entries
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check if line contains a compound
            compound_match = self._find_compound_in_line(line)
            if compound_match:
                # Extract numeric values
                values = self._extract_values_from_line(line, compound_match)
                
                if values:
                    # Determine which species/samples these values belong to
                    species_mapping = self._determine_species_context(lines, i)
                    
                    # Create compound entries
                    for species, value_data in self._assign_values_to_species(values, species_mapping).items():
                        if value_data['value'] is not None:
                            compounds.append({
                                'compound_name': compound_match,
                                'species': species,
                                'organism': 'seeds',  # Default for seed oil papers
                                'amount_value': value_data['value'],
                                'amount_unit': value_data['unit']
                            })
        
        return compounds
    
    def _find_compound_in_line(self, line: str) -> Optional[str]:
        """Find compound name in a line"""
        for pattern in self.compound_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return self._normalize_compound_name(match.group())
        
        # Also check for simple patterns like "Name number number number"
        parts = line.split()
        if len(parts) > 2:
            # Check if first part(s) could be a compound name
            potential_name = []
            for part in parts:
                if re.match(r'^\d+\.?\d*$', part):  # Hit a number
                    break
                potential_name.append(part)
            
            if potential_name and len(potential_name) <= 3:
                name = ' '.join(potential_name)
                # Check if it looks like a compound
                if any(keyword in name.lower() for keyword in ['acid', 'ol', 'ster', 'ine', 'ose']):
                    return self._normalize_compound_name(name)
        
        return None
    
    def _extract_values_from_line(self, line: str, compound_name: str) -> List[Tuple[float, str]]:
        """Extract numeric values and units from a line"""
        # Remove compound name to focus on values
        value_part = line.replace(compound_name, '', 1).strip()
        
        # Find all numeric patterns
        # Patterns: "12.3", "12.3±0.5", "ND", "tr" (trace)
        value_pattern = r'(?:(?:\d+\.?\d*(?:±\d+\.?\d*)?)|(?:ND)|(?:tr))'
        matches = re.findall(value_pattern, value_part)
        
        values = []
        for match in matches:
            if match == 'ND':
                values.append((0.0, 'ND'))
            elif match == 'tr':
                values.append((0.01, 'trace'))  # Use small value for trace
            else:
                # Extract main value (ignore ± part)
                main_value = float(re.match(r'(\d+\.?\d*)', match).group(1))
                values.append((main_value, ''))
        
        return values
    
    def _determine_species_context(self, lines: List[str], current_line_idx: int) -> Dict[str, str]:
        """Determine which species the values correspond to"""
        # Look for headers or context clues
        context_window = 10  # Look within 10 lines above
        
        # Default mapping for 3-species papers
        default_mapping = {
            0: 'Aronia melanocarpa L',  # Chokeberry
            1: 'Ribes nigrum L',         # Black currant
            2: 'Rosa canina L'           # Rose hip
        }
        
        # Look for explicit species mentions nearby
        for i in range(max(0, current_line_idx - context_window), current_line_idx):
            line_lower = lines[i].lower()
            for species_key, patterns in self.species_patterns.items():
                if any(pattern.lower() in line_lower for pattern in patterns):
                    # Found species context
                    self.logger.debug(f"Found species context: {species_key} at line {i}")
        
        # For now, return default mapping
        # In a real implementation, this would be more sophisticated
        return default_mapping
    
    def _assign_values_to_species(self, values: List[Tuple[float, str]], 
                                  species_mapping: Dict[int, str]) -> Dict[str, Dict]:
        """Assign values to species based on position"""
        result = {}
        
        # Determine unit based on value magnitude
        def guess_unit(value: float) -> str:
            if value > 100:
                return "g/kg oil"
            elif value > 10:
                return "mg/kg"
            elif value > 1:
                return "g/kg oil"
            else:
                return "mg/kg"
        
        # Common patterns:
        # - 3 values: one per species (chokeberry, blackcurrant, rosehip)
        # - 6 values: free and esterified for each species
        
        if len(values) == 3:
            # One value per species
            for i, (value, unit_hint) in enumerate(values):
                if i in species_mapping:
                    species = species_mapping[i]
                    unit = guess_unit(value) if not unit_hint else unit_hint
                    result[species] = {'value': value, 'unit': unit}
        
        elif len(values) == 6:
            # Two values per species (free and esterified)
            # Take the first value of each pair
            for i in range(0, 6, 2):
                species_idx = i // 2
                if species_idx in species_mapping:
                    species = species_mapping[species_idx]
                    value, unit_hint = values[i]
                    unit = guess_unit(value) if not unit_hint else unit_hint
                    result[species] = {'value': value, 'unit': unit}
        
        else:
            # Default: assign all values to first species
            # This is a fallback - in practice, we'd need more context
            if values and 0 in species_mapping:
                value, unit_hint = values[0]
                unit = guess_unit(value) if not unit_hint else unit_hint
                result[species_mapping[0]] = {'value': value, 'unit': unit}
        
        return result
    
    def _normalize_compound_name(self, name: str) -> str:
        """Normalize compound names"""
        # Replace common variations
        replacements = {
            'a-': 'α-',
            'b-': 'β-',
            'g-': 'γ-',
            'd-': 'δ-',
            'alpha-': 'α-',
            'beta-': 'β-',
            'gamma-': 'γ-',
            'delta-': 'δ-',
        }
        
        normalized = name.strip()
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Capitalize first letter
        if normalized and normalized[0].islower():
            normalized = normalized[0].upper() + normalized[1:]
        
        return normalized
    
    def extract_from_text_tables(self, text: str, paper_type: str = 'analytical') -> List[Dict]:
        """
        Main method to extract compounds from text-based tables
        
        Args:
            text: Full PDF text
            paper_type: Type of paper (analytical, isolation, etc.)
            
        Returns:
            List of compound dictionaries
        """
        self.logger.info(f"Parsing text-based tables for {paper_type} paper")
        
        if paper_type == 'analytical':
            compounds = self.parse_analytical_table(text)
        else:
            # For isolation papers, we might need different logic
            compounds = []
        
        self.logger.info(f"Extracted {len(compounds)} compounds from text tables")
        return compounds

# Test function
def test_parser():
    """Test the parser with sample data"""
    parser = CustomTableParser()
    
    # Sample text from the debug output
    sample_text = """
    Cholesterol 0.6 5.3 0.9 4.1 0.5 4.4
    Campesterol 6.0 5.0 1.5 1.0 1.8 1.2
    Brassicasterol ND ND ND ND 5.4 4.6
    Stigmasterol 2.6 5.1 6.8 3.0 3.5 1.6
    b-Sitosterol 89.8 73.8 87.3 85.7 81.5 81.6
    """
    
    compounds = parser.parse_analytical_table(sample_text)
    
    print(f"Found {len(compounds)} compounds:")
    for comp in compounds:
        print(f"  - {comp['compound_name']}: {comp['amount_value']} {comp['amount_unit']} "
              f"from {comp['species']}")

if __name__ == "__main__":
    test_parser()