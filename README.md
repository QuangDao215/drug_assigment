# LLM-Based Compound Extraction Pipeline

An automated pipeline for extracting chemical compound information from research PDFs using GPT-4 API (implemented with Groq for cost efficiency).

## 🎯 Overview

This pipeline extracts structured compound data from chemistry research papers, including:
- Compound names
- Species/source organisms  
- Amounts with units
- Confidence scores

## 📊 Performance Metrics

| Metric | Paper 1 | Paper 2 | Average |
|--------|---------|---------|---------|
| Precision | 0.823 | 0.606 | 0.715 |
| Recall | 0.765 | 0.606 | 0.686 |
| F1 Score | 0.793 | 0.606 | 0.700 |

*Based on evaluation against annotated ground truth data*

## 🏗️ Architecture

```
PDF Input → PDF Processor → Text Chunks → LLM Extractor → Compounds
                ↓                              ↓
        Custom Table Parser          Confidence Scoring
                ↓                              ↓
        Merged Results ← ← ← ← ← ← Validation System
                ↓
        Evaluation (for annotated) → Metrics & Reports
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key (or OpenAI API key with code modification)

### Installation

```bash
# Clone repository
git clone https://github.com/QuangDao215/drug_assigment.git
cd drug_assigment

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Running the Pipeline

```bash
# Process all PDFs
python run_full_pipeline.py

# Or process a single PDF
python run_pipeline.py --pdf_dir data/pdfs --output_dir data/outputs
```

## 📁 Project Structure

```
compound-extraction-pipeline/
├── src/
│   ├── __init__.py
│   ├── main_pipeline.py         # Main orchestration
│   ├── pdf_processor.py         # PDF text extraction
│   ├── text_chunker.py          # Smart text chunking
│   ├── groq_extractor.py        # LLM extraction
│   ├── custom_table_parser.py   # Table parsing
│   ├── targeted_table_parser.py # Enhanced table parsing
│   ├── paper_classifier.py      # Paper type detection
│   ├── prompt_templates.py      # LLM prompts
│   ├── data_models.py           # Data structures
│   ├── evaluator.py             # Evaluation metrics
│   ├── validator.py             # Validation system
│   └── ground_truth_extractor.py # Ground truth data
├── data/
│   ├── pdfs/                    # Input PDFs
│   └── outputs/                 # Results
│       ├── all_extracted_compounds.csv
│       ├── extraction_results.json
│       ├── performance_report.md
│       └── validation_reports/
├── notebooks/
│   └── test_pipeline.ipynb      # Testing notebook
├── debug/                       # Debug scripts
├── requirements.txt
├── .env.example
├── run_full_pipeline.py         # Main script
└── README.md

```

## 🔧 Key Features

### 1. Multi-Strategy Extraction
- **LLM-based extraction** using Groq (Llama 3.1)
- **Custom table parsing** for analytical chemistry papers
- **Paper type classification** for optimized prompts

### 2. Validation System
- Automated quality checks
- Confidence scoring
- Duplicate detection
- Range validation

### 3. Evaluation Framework
- Precision, Recall, F1 metrics
- Comparison with ground truth
- Detailed error analysis

### 4. Performance Optimization
- Batch processing
- Smart chunking
- Rate limiting
- Error recovery


## 🔄 Validation Approach

1. **Automated Checks**:
   - Required field validation
   - Format validation (species names, units)
   - Range validation (reasonable amounts)
   - Pattern detection (suspicious duplicates)

2. **Confidence Scoring**:
   - Based on extraction context
   - Section type (results vs methods)
   - Validation issues
   - Completeness of data

## 📝 Notes

- Currently uses Groq API (Llama 3.1) for cost efficiency
- Can be adapted to use OpenAI GPT-4 by modifying `groq_extractor.py`
- Optimized for chemistry papers with isolation and analytical data
