from src.data_models import Compound, Species, Organism, Amount, ExtractionResult
import json
import re

class GroundTruthExtractor:
    def __init__(self):
        self.extraction_results = {}
        self._load_ground_truth()
    
    def _create_compounds_batch(self, template: dict, compounds_data: list) -> list:
        """Create multiple compounds using a template with defaults"""
        compounds = []
        
        for data in compounds_data:
            # Merge template with specific data
            compound_data = template.copy()
            
            if isinstance(data, tuple):
                # Simple format: (name, amount_value)
                compound_data.update({
                    'name': data[0],
                    'amount_value': data[1]
                })
            elif isinstance(data, dict):
                # Override template with specific values
                compound_data.update(data)
            
            # Parse amount
            value = compound_data['amount_value']
            unit = compound_data.get('amount_unit', template.get('amount_unit', 'mg'))
            
            compound = Compound(
                name=compound_data['name'],
                species=Species(scientific_name=compound_data['species']),
                organism=Organism(name=compound_data['organism']),
                amount=Amount(value=value, unit=unit)
            )
            compounds.append(compound)
        
        return compounds
    
    def _load_ground_truth(self):
        """Load ground truth data using templates"""
        
        # Paper 1 - Ajuga bracteosa compounds
        paper1_result = ExtractionResult(paper_id="paper1")
        
        # Template for Ajuga bracteosa aerial parts compounds
        ajuga_template = {
            'species': 'Ajuga bracteosa',
            'organism': 'Aerial parts',
            'amount_unit': 'mg'
        }
        
        # Simple format: just name and amount value
        ajuga_compounds = [
            ('Ajubractin A', 4.5),
            ('Ajubractin B', 2.9),
            ('Ajubractin C', 14.7),
            ('Ajubractin D', 3.1),
            ('Ajubractin E', 0.9),
            ('15-Hydroxyajubractin C', 3.1),
            ('15-epi-Lupulin B', 2.0),
            ('Clerodin', 0.6),
            ('3-epi-Caryoptin', 36),
            ('Ajugapitin', 5.7),
            ('14,15-Dihydroclerodin', 42.1),
            ('3-epi-14,15-Dihydrocaryoptin', 4.6),
            ('Ivain II', 3.8),
            ('14,15-Dihydroajugapitin', 10.0),
            ('14-hydro-15-hydroxyajugachin A', 1.6),
            ('3β-hydroxydihydroclerodin', 0.9),
            ('14-hydro-15-hydroxyajugapitin', 3.8),
        ]
        
        compounds = self._create_compounds_batch(ajuga_template, ajuga_compounds)
        for compound in compounds:
            paper1_result.add_compound(compound)
        
        # Paper 2 - Multiple species
        paper2_result = ExtractionResult(paper_id="paper2")
        
        paper2_groups = [
            # Aronia melanocarpa L - fatty acids and sterols (g/kg oil)
            {
                'template': {
                    'species': 'Aronia melanocarpa L',
                    'organism': 'seeds',
                    'amount_unit': 'g/kg oil'
                },
                'compounds': [
                    ('Linoleic acid', 71.2),
                    ('Oleic acid', 21.4),
                    ('Palmitic acid', 5.1),
                    ('Stearic acid', 1.1),
                    ('Campesterol', 11),
                    ('Stigmasterol', 7.7),
                    ('β-Sitosterol', 163.6),
                    ('γ-Tocopherol', 28.2),  # Note: This unit seems inconsistent with other tocopherols
                ]
            },
            # Aronia melanocarpa L - tocopherols (mg/kg)
            {
                'template': {
                    'species': 'Aronia melanocarpa L',
                    'organism': 'seeds',
                    'amount_unit': 'mg/kg'
                },
                'compounds': [
                    ('α-Tocopherol', 70.6),
                    ('β-Tocopherol', 0.2),
                ]
            },
            # Ribes nigrum L - fatty acids and sterols (g/kg oil)
            {
                'template': {
                    'species': 'Ribes nigrum L',
                    'organism': 'seeds',
                    'amount_unit': 'g/kg oil'
                },
                'compounds': [
                    ('Linoleic acid', 57.8),
                    ('Oleic acid', 16.1),  # Note: Original had comma (16,1)
                    ('Palmitic acid', 6.4),
                    ('Stearic acid', 1.6),
                    ('Campesterol', 2.5),   # Note: Original had comma (2,5)
                    ('Stigmasterol', 9.8),
                    ('β-Sitosterol', 173),
                ]
            },
            # Ribes nigrum L - tocopherols (mg/kg)
            {
                'template': {
                    'species': 'Ribes nigrum L',
                    'organism': 'seeds',
                    'amount_unit': 'mg/kg'
                },
                'compounds': [
                    ('α-Tocopherol', 36.9),
                    ('γ-Tocopherol', 55.4),
                ]
            },
            # Rosa canina L - fatty acids and sterols (g/kg oil)
            {
                'template': {
                    'species': 'Rosa canina L',
                    'organism': 'seeds',
                    'amount_unit': 'g/kg oil'
                },
                'compounds': [
                    ('Linoleic acid', 2.1),
                    ('Oleic acid', 52.6),
                    ('Palmitic acid', 17.8),
                    ('Campesterol', 3),
                    ('Stigmasterol', 5.1),
                    ('β-Sitosterol', 163.1),
                ]
            },
            # Rosa canina L - stearic acid with different unit (g/kg)
            {
                'template': {
                    'species': 'Rosa canina L',
                    'organism': 'seeds',
                    'amount_unit': 'g/kg'
                },
                'compounds': [
                    ('Stearic acid', 8.8),
                ]
            },
            # Rosa canina L - tocopherols (mg/kg)
            {
                'template': {
                    'species': 'Rosa canina L',
                    'organism': 'seeds',
                    'amount_unit': 'mg/kg'
                },
                'compounds': [
                    ('α-Tocopherol', 19.0),
                    ('γ-Tocopherol', 71.0),
                    ('δ-Tocopherol', 6.9),  # Note: This was listed under Rosa canina L in the data
                ]
            },
        ]
        
        for group in paper2_groups:
            compounds = self._create_compounds_batch(group['template'], group['compounds'])
            for compound in compounds:
                paper2_result.add_compound(compound)
        
        self.extraction_results = {
            "paper1": paper1_result,
            "paper2": paper2_result
        }

    def get_paper_results(self, paper_id: str) -> ExtractionResult:
        """Get extraction results for a specific paper"""
        return self.extraction_results.get(paper_id)

    def save_to_json(self, filepath: str):
        """Save ground truth to JSON file"""
        data = {}
        for paper_id, result in self.extraction_results.items():
            data[paper_id] = {
                'compounds': [c.to_dict() for c in result.compounds]
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)