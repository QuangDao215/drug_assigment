from dataclasses import dataclass, field
from typing import List, Optional, Dict
import re
from enum import Enum

class UnitType(Enum):
    """Standardized unit types"""
    MASS = "mass"
    MASS_PER_MASS = "mass_per_mass"
    MASS_PER_VOLUME = "mass_per_volume"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"

@dataclass
class Amount:
    """Represents the amount of a compound with unit handling"""
    value: float
    unit: str
    unit_type: UnitType = field(init=False)
    normalized_unit: str = field(init=False)
    
    def __post_init__(self):
        self.unit_type = self._classify_unit()
        self.normalized_unit = self._normalize_unit()
    
    def _classify_unit(self) -> UnitType:
        """Classify the unit type"""
        unit_lower = self.unit.lower()
        
        if any(x in unit_lower for x in ['mg', 'g', 'kg', 'μg', 'ug']):
            if '/' in unit_lower:
                if 'kg' in unit_lower.split('/')[1]:
                    return UnitType.MASS_PER_MASS
                elif any(x in unit_lower for x in ['ml', 'l']):
                    return UnitType.MASS_PER_VOLUME
            else:
                return UnitType.MASS
        elif '%' in unit_lower:
            return UnitType.PERCENTAGE
        
        return UnitType.UNKNOWN
    
    def _normalize_unit(self) -> str:
        """Normalize units to standard forms"""
        unit_mapping = {
            'ug': 'μg',
            'microgram': 'μg',
            'milligram': 'mg',
            'gram': 'g',
            'kilogram': 'kg'
        }
        
        normalized = self.unit.lower()
        for old, new in unit_mapping.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def to_mg(self) -> Optional[float]:
        """Convert to milligrams if possible"""
        if self.unit_type != UnitType.MASS:
            return None
        
        conversions = {
            'μg': 0.001,
            'mg': 1.0,
            'g': 1000.0,
            'kg': 1000000.0
        }
        
        base_unit = self.normalized_unit.replace(' ', '')
        if base_unit in conversions:
            return self.value * conversions[base_unit]
        
        return None

@dataclass
class Organism:
    """Represents the organism/part where compound was found"""
    name: str
    normalized_name: str = field(init=False)
    
    def __post_init__(self):
        self.normalized_name = self._normalize()
    
    def _normalize(self) -> str:
        """Normalize organism names"""
        # Convert to lowercase, strip whitespace
        normalized = self.name.lower().strip()
        
        # Common normalizations
        replacements = {
            'aerial part': 'aerial parts',
            'seed': 'seeds',
            'leaf': 'leaves',
            'root': 'roots'
        }
        
        for old, new in replacements.items():
            if normalized == old:
                normalized = new
                
        return normalized

@dataclass
class Species:
    """Represents a plant species"""
    scientific_name: str
    common_name: Optional[str] = None
    normalized_name: str = field(init=False)
    
    def __post_init__(self):
        self.normalized_name = self._normalize_scientific_name()
    
    def _normalize_scientific_name(self) -> str:
        """Normalize species names (handle L., var., etc.)"""
        # Remove author abbreviations like L., Mill., etc.
        normalized = re.sub(r'\s+[A-Z][a-z]{0,3}\.?$', '', self.scientific_name)
        return normalized.strip()

@dataclass
class Compound:
    """Represents a chemical compound extracted from a plant"""
    name: str
    species: Species
    organism: Organism
    amount: Amount
    paper_id: Optional[str] = None
    confidence_score: float = 1.0
    
    def __str__(self):
        return (f"{self.name} from {self.species.scientific_name} "
                f"({self.organism.name}): {self.amount.value} {self.amount.unit}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'compound_name': self.name,
            'species': self.species.scientific_name,
            'species_normalized': self.species.normalized_name,
            'organism': self.organism.name,
            'organism_normalized': self.organism.normalized_name,
            'amount_value': self.amount.value,
            'amount_unit': self.amount.unit,
            'amount_unit_type': self.amount.unit_type.value,
            'paper_id': self.paper_id,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Compound':
        """Create compound from dictionary"""
        species = Species(scientific_name=data['species'])
        organism = Organism(name=data['organism'])
        amount = Amount(value=float(data['amount_value']), unit=data['amount_unit'])
        
        return cls(
            name=data['compound_name'],
            species=species,
            organism=organism,
            amount=amount,
            paper_id=data.get('paper_id'),
            confidence_score=data.get('confidence_score', 1.0)
        )

@dataclass
class ExtractionResult:
    """Container for all compounds extracted from a paper"""
    paper_id: str
    compounds: List[Compound] = field(default_factory=list)
    
    def add_compound(self, compound: Compound):
        """Add a compound to the results"""
        compound.paper_id = self.paper_id
        self.compounds.append(compound)
    
    def get_unique_species(self) -> List[Species]:
        """Get all unique species in this paper"""
        species_dict = {}
        for compound in self.compounds:
            key = compound.species.normalized_name
            if key not in species_dict:
                species_dict[key] = compound.species
        return list(species_dict.values())
    
    def get_compounds_by_species(self, species_name: str) -> List[Compound]:
        """Get all compounds from a specific species"""
        return [c for c in self.compounds 
                if species_name.lower() in c.species.normalized_name.lower()]
    
    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd
        data = [compound.to_dict() for compound in self.compounds]
        return pd.DataFrame(data)