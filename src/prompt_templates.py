class PromptTemplates:
    
    @staticmethod
    def get_system_prompt(paper_type: str) -> str:
        if paper_type == 'isolation_chemistry':
            return """You are a specialized chemistry AI that extracts compound isolation data.
Focus on compounds that were ACTUALLY ISOLATED with specific yields."""
        
        elif paper_type == 'analytical_chemistry':
            return """You are a specialized chemistry AI that extracts compound composition data.
Focus on quantitative analysis results showing compound amounts in samples."""
        
        return "You are a chemistry AI that extracts compound information."
    
    @staticmethod
    def get_user_prompt(text_chunk: str, paper_type: str) -> str:
        if paper_type == 'analytical_chemistry':
            return f"""Extract ALL compounds with quantitative data from this text. Be VERY inclusive.

EXTRACT anything that looks like:
- [Compound name] + [number] + [unit]
- Table rows with compound names and values
- Any chemical name followed by quantities

ACCEPTABLE PATTERNS:
✓ "Linoleic acid | 71.2 | g/kg oil" → Extract it
✓ "Oleic acid ranged from 16.1 to 52.6" → Extract highest value
✓ "α-Tocopherol (70.6 mg/kg)" → Extract it
✓ "Palmitic acid 5.1" → Extract even without clear units
✓ Table data even if poorly formatted

For tables, extract EVERY row that has:
- A compound/chemical name
- A numerical value
- Even if units are in header or unclear

If species is unclear, use the most recently mentioned species.
If units are unclear but values are present, guess reasonable units based on context.

Text:
{text_chunk}

Return ALL possible compounds as JSON array. When in doubt, INCLUDE it."""