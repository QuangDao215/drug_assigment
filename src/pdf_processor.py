# src/pdf_processor.py

import pdfplumber
import logging
from typing import List, Dict, Optional
from src.custom_table_parser import CustomTableParser
from src.targeted_table_parser import TargetedTableParser

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.custom_parser = CustomTableParser()
        self.targeted_parser = TargetedTableParser()
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF with enhanced table extraction"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                tables_found = 0
                custom_parsing_needed = False
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract regular text first
                    page_text = page.extract_text() or ""
                    
                    # Try to extract tables
                    tables = page.extract_tables()
                    
                    if tables:
                        self.logger.info(f"Found {len(tables)} table(s) on page {page_num + 1}")
                        tables_found += len(tables)
                        
                        # Add tables to the text
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                # Add table marker
                                page_text += f"\n\n[TABLE {table_idx + 1} - PAGE {page_num + 1}]\n"
                                
                                # Process each row
                                for row_idx, row in enumerate(table):
                                    if row:  # Skip empty rows
                                        # Clean cells and join with delimiter
                                        cleaned_cells = []
                                        for cell in row:
                                            if cell is not None:
                                                # Clean up cell content
                                                cell_text = str(cell).strip().replace('\n', ' ')
                                                cleaned_cells.append(cell_text)
                                            else:
                                                cleaned_cells.append("")
                                        
                                        # Join cells with a clear delimiter
                                        row_text = " | ".join(cleaned_cells)
                                        page_text += row_text + "\n"
                                
                                page_text += "[END TABLE]\n\n"
                    else:
                        # Check if this page might have text-based tables
                        if self._might_have_text_tables(page_text):
                            custom_parsing_needed = True
                            self.logger.info(f"Page {page_num + 1} might have text-based tables")
                    
                    full_text += page_text + "\n\n"
                
                # If no proper tables found but text tables detected, add marker
                if tables_found == 0 and custom_parsing_needed:
                    self.logger.warning("No structured tables found, but text-based tables detected")
                    # Add a marker for the extractor to know custom parsing is needed
                    full_text = "[CUSTOM_TABLE_PARSING_NEEDED]\n\n" + full_text
                
                # Log extraction summary
                self.logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
                self.logger.info(f"Structured tables found: {tables_found}")
                
                return full_text
                
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def _might_have_text_tables(self, text: str) -> bool:
        """Check if text might contain table-like data"""
        # Indicators of text-based tables
        indicators = [
            # Multiple numbers in a line
            lambda t: len([line for line in t.split('\n') 
                          if sum(1 for char in line if char.isdigit()) > 10]) > 5,
            
            # Lines with compound-like names followed by numbers
            lambda t: any(all(keyword in line.lower() 
                            for keyword in [word for word in ['acid', 'ol', 'ster'] 
                                          if word in line.lower()])
                         and any(char.isdigit() for char in line)
                         for line in t.split('\n')),
            
            # Presence of "ND" or "tr" (common in analytical tables)
            lambda t: ' ND ' in t or ' tr ' in t,
            
            # Multiple lines with similar structure (number patterns)
            lambda t: self._has_repeated_numeric_patterns(t)
        ]
        
        return any(indicator(text) for indicator in indicators)
    
    def _has_repeated_numeric_patterns(self, text: str) -> bool:
        """Check for repeated numeric patterns indicating tabular data"""
        import re
        
        lines = text.split('\n')
        numeric_lines = 0
        
        # Pattern for lines with multiple numbers
        pattern = re.compile(r'.*?\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*.*')
        
        for line in lines:
            if pattern.match(line):
                numeric_lines += 1
        
        # If more than 5 lines have this pattern, likely a table
        return numeric_lines > 5
    
    def extract_compounds_from_text_tables(self, text: str, paper_type: str = 'analytical') -> List[Dict]:
        """
        Extract compounds from text-based tables when structured extraction fails
        
        Args:
            text: PDF text
            paper_type: Type of paper
            
        Returns:
            List of compound dictionaries
        """
        # Use targeted parser for better accuracy
        return self.targeted_parser.parse_tables(text)