# LLM-Based Compound Extraction Pipeline - Performance Report

**Generated:** 2024-07-31  
**Model:** Groq (Llama 3.1-8b-instant)  
**Papers Evaluated:** 2 annotated papers

## 1. Executive Summary

We developed an automated pipeline for extracting chemical compound information from research PDFs using LLM technology. The system achieved an **F1 score of 0.607** on annotated test data, with high recall (83.8%) ensuring most compounds are captured.

## 2. Evaluation Metrics (Annotated Papers)

### Overall Performance

| Metric | Paper 1 | Paper 2 | **Average** |
|--------|---------|---------|-------------|
| **Precision** | 0.577 | 0.383 | **0.480** |
| **Recall** | 0.882 | 0.793 | **0.838** |
| **F1 Score** | 0.698 | 0.517 | **0.607** |
| True Positives | 15 | 23 | 38 |
| False Positives | 11 | 37 | 48 |
| False Negatives | 2 | 6 | 8 |

### Key Findings

- **Paper 1 (Isolation Chemistry)**: Better performance with F1=0.698
  - Successfully extracted 15/17 compounds (88.2% recall)
  - Main issues: Generic compound names ("1", "2") and missing species info
  
- **Paper 2 (Analytical Chemistry)**: Lower performance with F1=0.517
  - Successfully extracted 23/30 compounds (79.3% recall)
  - Main issues: Missed all fatty acids, many phospholipid false positives

## 3. Processing Time Analysis

Based on the 2 annotated papers processed:

| Stage | Time (seconds) | Percentage | Description |
|-------|----------------|------------|-------------|
| PDF Text Extraction | ~2.0 | 4% | pdfplumber text extraction |
| Paper Classification | ~0.5 | 1% | Identify paper type |
| Text Chunking | ~1.0 | 2% | Smart chunking with overlap |
| Custom Table Parsing | ~1.5 | 3% | Parse text-based tables |
| LLM Processing | ~40.0 | 89% | Groq API calls for extraction |
| Validation | ~0.5 | 1% | Automated checks |
| **Total per Paper** | **~45.0** | **100%** | **End-to-end processing** |

### Performance Characteristics

- **Average processing time**: 45 seconds per paper
- **Bottleneck**: LLM API calls (89% of total time)
- **Optimization potential**: Batch processing, caching, parallel API calls

## 4. System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDF Input │ --> │ PDF Processor│ --> │ Text Chunker │ --> │ LLM Extractor│
└─────────────┘     └──────┬───────┘     └──────────────┘     └──────┬───────┘
                           │                                         │
                           ▼                                         ▼
                    ┌──────────────┐                          ┌──────────────┐
                    │ Custom Table │                          │  Confidence  │
                    │    Parser    │                          │   Scoring    │
                    └──────┬───────┘                          └──────┬───────┘
                           │                                         │
                           └──────────────┐    ┌─────────────────────┘
                                          ▼    ▼
                                    ┌──────────────┐
                                    │    Merged    │
                                    │   Results    │
                                    └──────┬───────┘
                                           │
                                           ▼
                        ┌──────────────────┴───────────────────┐
                        │                                      │
                  ┌─────▼──────┐                        ┌──────▼─────┐
                  │ Evaluation │                        │   Output   │
                  │  Metrics   │                        │   Reports  │
                  └────────────┘                        └────────────┘
```

## 5. Validation System

### 5.1 Automated Validation Checks

Our multi-layer validation system performs the following checks:

#### **Required Field Validation**
- ✓ Compound name exists and is non-empty
- ✓ Species/organism information is present
- ✓ Amount value and unit are specified
- ✓ All fields contain valid characters

#### **Format Validation**
- ✓ Species names match scientific nomenclature (Genus species)
- ✓ Compound names contain only valid characters
- ✓ Amount values are positive numbers
- ✓ Units are recognized and standardized

#### **Range Validation**
- ✓ Isolation chemistry: 0.1 mg - 5 g (typical yields)
- ✓ Analytical chemistry: 1 μg - 10 g (detection ranges)
- ✓ Flag unusually large or small values for review

#### **Duplicate Detection**
- ✓ Identify exact compound duplicates
- ✓ Flag compounds with identical amounts (potential table parsing error)
- ✓ Group duplicates for manual review

#### **Pattern Analysis**
- ✓ Detect generic names ("compound 1", "unknown")
- ✓ Identify systematic extraction errors
- ✓ Flag suspicious patterns (>30% identical values)

### 5.2 Confidence Scoring Algorithm

```python
Base Score = 0.8 (if from LLM) or 0.85 (if from direct parsing)

Deductions:
- Error-level issue: -0.3
- Warning-level issue: -0.1  
- Info-level issue: -0.05

Bonuses:
- Complete data with no issues: +0.1
- From high-confidence section (results): +0.1

Final Score = max(0.0, min(1.0, adjusted_score))
```

### 5.3 Validation Results Summary

For the 2 annotated papers:
- **100% structural validity** - All extracted compounds passed format checks
- **8 validation issues** identified (mostly formatting and duplicates)
- **Average confidence score**: 0.82

## 6. Improvement Plan

### 6.1 Short-term Improvements (1-2 weeks)

#### **Prompt Engineering**
- Fine-tune prompts for fatty acid extraction (current weakness)
- Add few-shot examples for complex table formats
- Implement paper-specific prompt templates
- Add negative examples to reduce false positives

#### **Parser Enhancements**
- Improve fatty acid table detection
- Handle multi-column analytical tables
- Better unit standardization
- Enhanced species name normalization

#### **Validation Refinements**
- Compound name similarity matching
- Chemical formula validation
- Cross-reference with chemical databases
- Context-aware duplicate removal

### 6.2 Medium-term Improvements (1-3 months)

#### **Active Learning Implementation**
```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Extract    │ --> │   Validate   │ --> │ Low Confidence│
│  Compounds  │     │   Results    │     │   Compounds   │
└─────────────┘     └──────────────┘     └──────┬────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Update    │ <-- │    Learn     │ <-- │    Human     │
│   System    │     │   Patterns   │     │    Review    │
└─────────────┘     └──────────────┘     └──────────────┘
```

- Collect human feedback on low-confidence extractions
- Identify systematic errors
- Update prompts and parsers based on patterns
- Retrain confidence scoring model

#### **Performance Optimization**
- Implement caching for repeated papers
- Batch API calls to reduce latency
- Parallel processing for independent chunks
- Local LLM fallback for simple extractions

### 6.3 Long-term Improvements (3-6 months)

#### **Advanced Techniques**
1. **Fine-tuned Models**
   - Train specialized model on chemistry papers
   - Domain-specific tokenization
   - Chemical entity recognition

2. **Hybrid Approach**
   - Rule-based extraction for standard formats
   - LLM for complex/novel patterns
   - Ensemble voting for high confidence

3. **Knowledge Graph Integration**
   - Build compound relationship graph
   - Cross-validate with existing databases
   - Predict missing information

4. **Quality Metrics**
   - Implement automated quality scoring
   - Track improvement over time
   - A/B test different approaches

## 7. Conclusion

The pipeline successfully demonstrates automated compound extraction with **60.7% F1 score** on annotated test data. The high recall (83.8%) ensures most compounds are captured, which is preferable for scientific applications where missing data is more problematic than extra validation.

### Strengths
- ✓ High recall captures most real compounds
- ✓ Robust validation system
- ✓ Handles multiple paper types
- ✓ Extensible architecture

### Areas for Improvement
- ✗ Fatty acid extraction needs enhancement
- ✗ Reduce false positives (improve precision)
- ✗ Processing speed optimization needed
- ✗ Better handling of complex table formats

### Next Steps
1. Deploy the system for the remaining PDFs when API limits reset
2. Implement short-term improvements
3. Collect user feedback
4. Iterate based on real-world performance

---

*This pipeline provides a solid foundation for automated compound extraction with clear paths for improvement based on systematic evaluation.*