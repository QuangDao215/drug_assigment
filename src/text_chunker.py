import re
from typing import List, Dict
import logging  # Add this import

class TextChunker:
    def __init__(self, chunk_size=2000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.paper_type = 'isolation_chemistry'  # Add default
        self.logger = logging.getLogger(__name__)  # Add logger

    def classify_chunk_section(self, chunk_text: str, paper_type: str = 'isolation_chemistry') -> str:
        """Classify chunk section based on paper type"""
        
        # Use instance paper_type if not provided
        if paper_type is None:
            paper_type = getattr(self, 'paper_type', 'isolation_chemistry')
        
        self.logger.info(f"Classifying chunk with paper type: {paper_type}")
        
        text_lower = chunk_text.lower()
        
        if paper_type == 'isolation_chemistry':
            # Your existing classification logic
            if any(phrase in text_lower for phrase in [
                'isolated', 'yielded', 'obtained', 'afforded',
                'isolation of', 'compounds 1-', 'compound 1,', 
                'fractionation', 'purification yielded'
            ]):
                return 'isolation_results'
        
            # Literature/comparison section
            elif any(phrase in text_lower for phrase in [
                'previously reported', 'literature', 'known compound',
                'compared with', 'reference', 'standard', 'authentic',
                'has been reported', 'was reported', 'previously isolated'
            ]):
                return 'literature_comparison'
            
            # Methodology section
            elif any(phrase in text_lower for phrase in [
                'extraction procedure', 'plant material', 'general experimental',
                'extracted with', 'dissolved in', 'chromatography conditions',
                'nmr spectra', 'mass spectra'
            ]):
                return 'methodology'
            
            # Introduction/background
            elif any(phrase in text_lower for phrase in [
                'traditional medicine', 'medicinal plant', 'widely distributed',
                'previous studies', 'biological activities', 'pharmacological'
            ]):
                return 'introduction'

        elif paper_type == 'analytical_chemistry':
            # New classification for analytical papers
            if any(phrase in text_lower for phrase in [
                'results and discussion', 'results', 
                'composition of', 'content of', 'analysis showed',
                'table', 'were determined', 'were found',
                'ranged from', 'varied from'
            ]):
                return 'analytical_results'
            
            elif any(phrase in text_lower for phrase in [
                'materials and methods', 'experimental',
                'extraction procedure', 'analytical methods',
                'gc conditions', 'hplc conditions'
            ]):
                return 'methodology'
            
            elif any(phrase in text_lower for phrase in [
                'compared with', 'similar to', 'literature',
                'previous studies', 'reported values'
            ]):
                return 'literature_comparison'
        
        return 'other'
    
    def create_chunks(self, text: str) -> List[Dict]:
        """Create overlapping chunks with metadata"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Classify and save current chunk
                section_type = self.classify_chunk_section(current_chunk)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'char_start': max(0, current_length - len(current_chunk)),
                    'char_end': current_length,
                    'section_type': section_type,
                    'priority': 'high' if section_type == 'isolation_results' else 'normal'
                })
                
                # Create overlap
                overlap_sentences = []
                temp_length = 0
                for j in range(i-1, -1, -1):
                    if temp_length < self.overlap:
                        overlap_sentences.insert(0, sentences[j])
                        temp_length += len(sentences[j])
                    else:
                        break
                
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_length += len(sentence)
        
        # Don't forget the last chunk
        if current_chunk:
            section_type = self.classify_chunk_section(current_chunk)
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'char_start': max(0, current_length - len(current_chunk)),
                'char_end': current_length,
                'section_type': section_type,
                'priority': 'high' if section_type == 'isolation_results' else 'normal'
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_smart_chunks(self, text: str) -> List[Dict]:
        """Create chunks with section awareness"""
        
        # Try to identify major sections first
        section_pattern = r'\n\s*(?:RESULTS?|METHODS?|MATERIALS?|DISCUSSION|INTRODUCTION|ABSTRACT|EXPERIMENTAL|ISOLATION)\s*\n'
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        all_chunks = []
        
        # If we found clear sections, process them separately
        if len(sections) > 1:
            for section in sections:
                if len(section.strip()) > 50:  # Skip very short sections
                    chunks = self.create_chunks(section)
                    all_chunks.extend(chunks)
        else:
            # No clear sections found, process as one text
            all_chunks = self.create_chunks(text)
        
        # Log section distribution for debugging
        section_counts = {}
        for chunk in all_chunks:
            section_type = chunk['section_type']
            section_counts[section_type] = section_counts.get(section_type, 0) + 1
        
        print(f"Chunk distribution by section: {section_counts}")
        
        return all_chunks