import json
import os
from typing import List, Dict, Optional
from groq import Groq
import time
from dotenv import load_dotenv
from src.data_models import Compound, Species, Organism, Amount, ExtractionResult
import re
import logging

# Load environment variables
load_dotenv()

class GroqExtractor:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq extractor
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        self.model = model
        
        self.logger.info(f"Initialized Groq extractor with model: {model}")

        self.paper_type = 'isolation_chemistry'  # default
        
        # Import prompt templates
        from src.prompt_templates import PromptTemplates
        self.prompt_templates = PromptTemplates()

    def create_system_prompt(self) -> str:
        """Create system prompt based on paper type"""
        return self.prompt_templates.get_system_prompt(self.paper_type)

    def create_user_prompt(self, text_chunk: str) -> str:
        """Create user prompt based on paper type"""
        self.logger.info(f"Creating prompt for paper type: {getattr(self, 'paper_type', 'NOT SET')}")
        
        # Check if we have prompt templates
        if hasattr(self, 'prompt_templates'):
            return self.prompt_templates.get_user_prompt(text_chunk, self.paper_type)
        else:
            self.logger.warning("Prompt templates not initialized!")
            # Fall back to old prompt
            return self.create_user_prompt_old(text_chunk)  # Your original prompt

    def create_system_prompt(self) -> str:
        """Create system prompt for extraction"""
        return """You are a specialized chemistry AI that extracts compound information from scientific papers.

CRITICAL DISTINCTION:
- ONLY extract compounds that were ACTUALLY ISOLATED IN THIS SPECIFIC STUDY
- DO NOT extract compounds that are mentioned for comparison, reference, or from previous literature
- DO NOT extract compounds from methodology sections (extraction solvents, reagents)
- Look for clear isolation language: "afforded", "yielded", "obtained", "isolated"

IMPORTANT: Each compound isolated in a real study will have its own unique yield. If you see multiple compounds with identical amounts, you're likely looking at a comparison table, not actual isolation results."""

    def create_user_prompt(self, text_chunk: str) -> str:
        """Create user prompt with few-shot examples"""
        return f"""Extract ONLY compounds that were actually isolated in THIS study.

CORRECT EXTRACTION EXAMPLES:
✓ "Fractionation of the EtOAc extract afforded ajubractin A (1, 4.5 mg) as white crystals"
  → {{"compound_name": "ajubractin A", "species": "Ajuga bracteosa", "organism": "aerial parts", "amount_value": 4.5, "amount_unit": "mg"}}

✓ "Compound 3, ajubractin C (14.7 mg), was obtained from fraction F4"
  → {{"compound_name": "ajubractin C", "species": "Ajuga bracteosa", "organism": "aerial parts", "amount_value": 14.7, "amount_unit": "mg"}}

✓ "Further purification yielded 15-hydroxyajubractin C (6, 3.1 mg)"
  → {{"compound_name": "15-hydroxyajubractin C", "species": "Ajuga bracteosa", "organism": "aerial parts", "amount_value": 3.1, "amount_unit": "mg"}}

INCORRECT EXTRACTION EXAMPLES (DO NOT EXTRACT):
✗ "Clerodin (7.83 mg/g) has been previously isolated from A. bracteosa [12]" 
  → SKIP: This is referencing previous work, not this study

✗ "The known compounds were identified by comparison with authentic samples"
  → SKIP: These are reference compounds, not isolated in this study

✗ "Table 1. Compounds reported from Ajuga species: Compound A (7.83 mg/g)..."
  → SKIP: This is a literature comparison table

✗ "The dried plant material (500 g) was extracted with MeOH"
  → SKIP: This is methodology, not an isolated compound

✗ "Standards of linoleic acid, oleic acid were used for comparison"
  → SKIP: These are reference standards

KEY PATTERNS TO LOOK FOR:
- Actual isolation: "afforded", "yielded", "obtained", "isolated as", "gave"
- With fractions: "from fraction F3", "from the hexane fraction"
- Physical description: "as white powder", "as yellow oil", "as colorless crystals"

WARNING SIGNS TO AVOID:
- "previously reported", "known compound", "has been isolated"
- "compared with", "identified as", "standard"
- Multiple compounds with identical amounts (e.g., all 7.83 mg/g)
- Extraction solvents or reagents

Text to analyze:
{text_chunk}

Return ONLY compounds actually isolated in this work as JSON array. If no compounds were isolated in this chunk, return []."""

    def extract_from_chunk(self, text_chunk: str, retry_count: int = 3) -> List[Dict]:
        """
        Extract compounds from a single text chunk
        
        Args:
            text_chunk: Text to analyze
            retry_count: Number of retries on failure
            
        Returns:
            List of compound dictionaries
        """
        
        for attempt in range(retry_count):
            try:
                # Create chat completion
                self.logger.info(f"Sending request to Groq API (attempt {attempt + 1})")
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": self.create_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": self.create_user_prompt(text_chunk)
                        }
                    ],
                    model=self.model,
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000,
                    top_p=0.9,
                )
                
                response_text = chat_completion.choices[0].message.content.strip()
                self.logger.info(f"Received response: {response_text[:200]}...")
                
                # Parse JSON response
                compounds = self._parse_json_response(response_text)
                
                # Validate compounds
                valid_compounds = self._validate_compounds(compounds)

                # Apply literature filter if we have valid compounds
                if valid_compounds and len(text_chunk) > 100:  # Only filter if we have enough context
                    valid_compounds = self.filter_literature_compounds(valid_compounds, text_chunk)

                self.logger.info(f"Extracted {len(valid_compounds)} valid compounds from chunk")
                return valid_compounds
                
            except Exception as e:
                self.logger.error(f"Extraction attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("All extraction attempts failed")
                    
        return []

    def validate_extraction_results(self, compounds: List[Dict]) -> List[Dict]:
        """Additional validation to reduce false positives"""
        
        validated = []
        seen_compounds = set()
        
        for comp in compounds:
            # Create unique key
            key = f"{comp['compound_name']}_{comp['species']}_{comp['organism']}"
            
            # Skip if we've seen this exact compound
            if key in seen_compounds:
                continue
                
            # Validate amount is reasonable
            amount = comp['amount_value']
            unit = comp['amount_unit'].lower()
            
            # Check for reasonable ranges
            if 'mg' in unit and amount > 10000:  # > 10g in mg is suspicious
                self.logger.warning(f"Suspicious amount: {amount} {unit}")
                continue
                
            # Ensure species name looks valid
            if len(comp['species']) < 5 or not any(c.isalpha() for c in comp['species']):
                self.logger.warning(f"Invalid species name: {comp['species']}")
                continue
                
            seen_compounds.add(key)
            validated.append(comp)
        
        return validated

    def _parse_json_response(self, response_text: str) -> List[Dict]:
        """Parse JSON from response text"""
        try:
            # First try direct parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON array from text
            self.logger.info("Direct JSON parsing failed, trying regex extraction")
            
            # Look for JSON array pattern
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse extracted JSON")
                    
            # Try to find JSON between code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse JSON from code block")
                    
        return []
    
    def _validate_compounds(self, compounds: List[Dict]) -> List[Dict]:
        """Validate and clean compound data with enhanced filtering"""
        valid_compounds = []
        required_fields = ['compound_name', 'species', 'organism']
        
        # Track amounts to detect suspicious patterns
        amount_frequency = {}
        
        for comp in compounds:
            # Check basic required fields
            if not all(field in comp for field in required_fields):
                self.logger.warning(f"Compound missing required fields: {comp}")
                continue
            
            # Handle amount field - it might come as combined 'amount' or separate fields
            if 'amount' in comp and isinstance(comp['amount'], str):
                # Parse combined amount like "4.5 mg"
                amount_match = re.match(r'([\d.]+)\s*(.+)', comp['amount'])
                if amount_match:
                    comp['amount_value'] = float(amount_match.group(1))
                    comp['amount_unit'] = amount_match.group(2).strip()
                    del comp['amount']
                else:
                    self.logger.warning(f"Could not parse amount: {comp['amount']}")
                    continue
            
            # Now check if we have the separated fields
            if 'amount_value' not in comp or 'amount_unit' not in comp:
                self.logger.warning(f"Missing amount_value or amount_unit: {comp}")
                continue
            
            # Validate and convert amount_value
            try:
                comp['amount_value'] = float(comp['amount_value'])
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid amount_value: {comp.get('amount_value')}")
                continue
            
            # Track amount frequency
            amount_key = f"{comp['amount_value']} {comp['amount_unit']}"
            amount_frequency[amount_key] = amount_frequency.get(amount_key, 0) + 1
            
            # Clean string fields
            comp['compound_name'] = str(comp['compound_name']).strip()
            comp['species'] = str(comp['species']).strip()
            comp['organism'] = str(comp['organism']).strip()
            comp['amount_unit'] = str(comp['amount_unit']).strip()
            
            # Skip if any field is empty or contains placeholder text
            if not all(comp.get(field) for field in ['compound_name', 'species', 'organism', 'amount_unit']):
                self.logger.warning(f"Compound has empty fields: {comp}")
                continue
            
            # Skip suspicious compound names
            suspicious_names = ['compound', 'extract', 'fraction', 'standard', 'reference']
            if any(susp in comp['compound_name'].lower() for susp in suspicious_names):
                self.logger.warning(f"Suspicious compound name: {comp['compound_name']}")
                continue
            
            valid_compounds.append(comp)
        
        # Check for suspicious patterns
        if amount_frequency:
            max_frequency = max(amount_frequency.values())
            if max_frequency > 3:
                most_common_amount = [k for k, v in amount_frequency.items() if v == max_frequency][0]
                self.logger.warning(f"Suspicious pattern: {max_frequency} compounds have identical amount ({most_common_amount})")
                self.logger.warning("This might indicate extraction from a comparison table rather than isolation results")
        
        return valid_compounds
    
    def extract_from_chunks(self, chunks: List[Dict], batch_size: int = 5) -> ExtractionResult:
        """Extract compounds from multiple chunks with section awareness"""
        
        all_compounds = []
        total_chunks = len(chunks)
        
        # Sort chunks by priority - process isolation_results sections first
        isolation_chunks = [c for c in chunks if c.get('section_type') == 'isolation_results']
        other_chunks = [c for c in chunks if c.get('section_type') != 'isolation_results']
        
        # Log the distribution
        self.logger.info(f"Found {len(isolation_chunks)} isolation chunks and {len(other_chunks)} other chunks")
        
        # Process isolation chunks first
        ordered_chunks = isolation_chunks + other_chunks
        
        for i, chunk in enumerate(ordered_chunks):
            self.logger.info(f"Processing chunk {i+1}/{total_chunks} (section: {chunk.get('section_type', 'unknown')})")
            
            # Add longer delays between requests
            if i > 0:
                time.sleep(2)  # 2 second delay between all requests
            
            # Extract compounds
            extracted_data = self.extract_from_chunk(chunk['text'])
            validated_data = self.validate_extraction_results(extracted_data)
            
            # Log extraction results by section
            if validated_data:
                self.logger.info(f"Extracted {len(validated_data)} compounds from {chunk.get('section_type', 'unknown')} section")
            
            # Convert to Compound objects
            for data in validated_data:
                try:
                    compound = Compound(
                        name=data['compound_name'],
                        species=Species(scientific_name=data['species']),
                        organism=Organism(name=data['organism']),
                        amount=Amount(
                            value=data['amount_value'],
                            unit=data['amount_unit']
                        ),
                        confidence_score=0.9 if chunk.get('section_type') == 'isolation_results' else 0.7
                    )
                    all_compounds.append(compound)
                    self.logger.info(f"Created compound: {compound}")
                    
                except Exception as e:
                    self.logger.error(f"Error creating compound object: {e}")
                    self.logger.error(f"Data: {data}")
            
            # Rate limiting - pause after batch
            if (i + 1) % batch_size == 0 and i < total_chunks - 1:
                self.logger.info(f"Processed {i+1} chunks, pausing for rate limit...")
                time.sleep(1)
        
        # Create extraction result
        result = ExtractionResult(paper_id="extracted")
        for compound in all_compounds:
            result.add_compound(compound)
        
        # Remove duplicates
        unique_result = self._deduplicate_compounds(result)
        
        self.logger.info(f"Total compounds extracted: {len(unique_result.compounds)}")
        return unique_result
    
    def _deduplicate_compounds(self, result: ExtractionResult) -> ExtractionResult:
        """Remove duplicate compounds based on key fields"""
        seen = set()
        unique_result = ExtractionResult(paper_id=result.paper_id)
        
        for compound in result.compounds:
            # Create unique key
            key = (
                compound.name.lower(),
                compound.species.normalized_name.lower(),
                compound.organism.normalized_name.lower(),
                compound.amount.value,
                compound.amount.normalized_unit.lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique_result.add_compound(compound)
            else:
                self.logger.info(f"Duplicate compound removed: {compound}")
        
        self.logger.info(f"Deduplication: {len(result.compounds)} -> {len(unique_result.compounds)} compounds")
        return unique_result

    def test_connection(self) -> bool:
        """Test if Groq API connection works"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Say 'Connection successful' and nothing else."}
                ],
                model=self.model,
                max_tokens=10
            )
            result = response.choices[0].message.content
            self.logger.info(f"Connection test: {result}")
            return "successful" in result.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
        

    def filter_literature_compounds(self, compounds: List[Dict], chunk_text: str) -> List[Dict]:
        """Filter out compounds that appear to be from literature/comparison sections"""
        
        filtered = []
        chunk_lower = chunk_text.lower()
        
        for comp in compounds:
            compound_name_lower = comp['compound_name'].lower()
            
            # Check if this compound appears in a literature context
            literature_patterns = [
                f"{compound_name_lower}.*previously",
                f"known.*{compound_name_lower}",
                f"{compound_name_lower}.*reported",
                f"{compound_name_lower}.*identified by comparison",
                f"authentic.*{compound_name_lower}",
                f"{compound_name_lower}.*standard"
            ]
            
            # Check if any literature pattern matches
            is_literature = any(re.search(pattern, chunk_lower) for pattern in literature_patterns)
            
            if not is_literature:
                filtered.append(comp)
            else:
                self.logger.info(f"Filtered out literature compound: {comp['compound_name']}")
        
        return filtered