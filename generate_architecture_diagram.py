# generate_architecture_diagram.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_architecture_diagram():
    """Create a visual architecture diagram for the pipeline"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    input_color = '#E8F4FD'
    process_color = '#B8E0D2'
    ml_color = '#D4A5A5'
    output_color = '#FFECB3'
    
    # Title
    ax.text(7, 9.5, 'LLM-Based Compound Extraction Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Stage 1: Input
    input_box = FancyBboxPatch((0.5, 7), 2.5, 1.5, 
                               boxstyle="round,pad=0.1",
                               facecolor=input_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 7.75, 'PDF Input\n(Research Papers)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Stage 2: PDF Processing
    pdf_box = FancyBboxPatch((4, 7), 2.5, 1.5,
                             boxstyle="round,pad=0.1", 
                             facecolor=process_color,
                             edgecolor='black', linewidth=2)
    ax.add_patch(pdf_box)
    ax.text(5.25, 7.75, 'PDF Processor\n(Text Extraction)', 
            ha='center', va='center', fontsize=10)
    
    # Stage 3: Text Chunking
    chunk_box = FancyBboxPatch((7.5, 7), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=process_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(chunk_box)
    ax.text(8.75, 7.75, 'Text Chunker\n(Smart Chunking)', 
            ha='center', va='center', fontsize=10)
    
    # Stage 4: LLM Processing
    llm_box = FancyBboxPatch((11, 7), 2.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=ml_color,
                             edgecolor='black', linewidth=2)
    ax.add_patch(llm_box)
    ax.text(12.25, 7.75, 'LLM Extractor\n(Groq/GPT-4)', 
            ha='center', va='center', fontsize=10)
    
    # Stage 5: Custom Parser (parallel)
    parser_box = FancyBboxPatch((4, 4.5), 2.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=process_color,
                                edgecolor='black', linewidth=2)
    ax.add_patch(parser_box)
    ax.text(5.25, 5.25, 'Custom Table\nParser', 
            ha='center', va='center', fontsize=10)
    
    # Stage 6: Validation
    valid_box = FancyBboxPatch((7.5, 4.5), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=ml_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(valid_box)
    ax.text(8.75, 5.25, 'Validation\n& Confidence', 
            ha='center', va='center', fontsize=10)
    
    # Stage 7: Merged Results
    merge_box = FancyBboxPatch((5.75, 2), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=process_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(merge_box)
    ax.text(7, 2.75, 'Merged\nResults', 
            ha='center', va='center', fontsize=10)
    
    # Stage 8: Evaluation
    eval_box = FancyBboxPatch((1, 2), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=ml_color,
                              edgecolor='black', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(2.5, 2.75, 'Evaluation\n(Precision/Recall/F1)', 
            ha='center', va='center', fontsize=10)
    
    # Stage 9: Output
    output_box = FancyBboxPatch((9.5, 2), 3.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=output_color,
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(11.25, 2.75, 'Reports & Data\n(CSV, JSON, MD)', 
            ha='center', va='center', fontsize=10)
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Main flow
    ax.annotate('', xy=(4, 7.75), xytext=(3, 7.75), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 7.75), xytext=(6.5, 7.75), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 7.75), xytext=(10, 7.75), arrowprops=arrow_props)
    
    # Parser branch
    ax.annotate('', xy=(5.25, 6.5), xytext=(5.25, 7), 
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # To validation
    ax.annotate('', xy=(8.75, 6), xytext=(12.25, 7), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red', 
                               connectionstyle="arc3,rad=-.3"))
    ax.annotate('', xy=(8.75, 6), xytext=(5.25, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='blue', 
                               connectionstyle="arc3,rad=.3"))
    
    # To merge
    ax.annotate('', xy=(7, 3.5), xytext=(8.75, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # To evaluation and output
    ax.annotate('', xy=(2.5, 3.5), xytext=(7, 2), 
                arrowprops=dict(arrowstyle='->', lw=2, color='purple',
                               connectionstyle="arc3,rad=.3"))
    ax.annotate('', xy=(9.5, 2.75), xytext=(8.25, 2.75), arrowprops=arrow_props)
    
    # Legend
    legend_elements = [
        mlines.Line2D([0], [0], color='black', lw=2, label='Main Flow'),
        mlines.Line2D([0], [0], color='blue', lw=2, label='Table Parsing'),
        mlines.Line2D([0], [0], color='red', lw=2, label='Validation'),
        mlines.Line2D([0], [0], color='green', lw=2, label='Merge'),
        mlines.Line2D([0], [0], color='purple', lw=2, label='Evaluation')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    # Add annotations
    ax.text(7, 0.5, 'Processing Time: ~45 seconds per paper | Accuracy: F1=0.70', 
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig('data/outputs/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/outputs/architecture_diagram.pdf', bbox_inches='tight')
    print("Architecture diagram saved to data/outputs/")
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()