# Text Mining at Scale for Taxonomy Generation

An automated taxonomy generation system that uses Large Language Models (LLMs) to create and iteratively refine taxonomies from large-scale text datasets. This project leverages LangChain and LangGraph to build an intelligent workflow that processes, summarizes, and categorizes documents at scale.

## Overview

This project implements an intelligent taxonomy generation pipeline that:
- Summarizes large volumes of documents using LLMs
- Generates initial taxonomies based on document content
- Iteratively refines taxonomies using minibatch processing
- Reviews and finalizes taxonomies for production use

The system is designed to handle large-scale text mining tasks and can be applied to various use cases such as customer feedback categorization, content classification, and document organization.

## Key Features

- **Automated Summarization**: Condenses lengthy documents into concise summaries and explanations
- **Scalable Processing**: Handles large datasets through intelligent minibatch processing
- **Iterative Refinement**: Continuously improves taxonomy quality through multiple update cycles
- **State Management**: Uses LangGraph for robust workflow orchestration
- **Tracing & Debugging**: Integrated with LangSmith for comprehensive monitoring
- **Flexible Configuration**: Highly customizable parameters for different use cases

## Architecture

The system uses a state graph architecture with the following workflow:

```
START → Summarize → Get Minibatches → Generate Taxonomy → Update Taxonomy ⟲ → Review Taxonomy → END
```

### Workflow Steps

1. **Summarize**: Processes documents in parallel to generate concise summaries and explanations
2. **Get Minibatches**: Splits documents into randomized batches for iterative processing
3. **Generate Taxonomy**: Creates initial taxonomy from the first minibatch
4. **Update Taxonomy**: Iteratively refines taxonomy using subsequent minibatches
5. **Review Taxonomy**: Final review and validation of the generated taxonomy

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- OpenAI API key
- Kaggle API credentials (for dataset downloads)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Azamat0315277/Text-mining-at-Scale-for-taxonomy.git
cd Text-mining-at-Scale-for-taxonomy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Create a .env file with the following:
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

4. Set up Kaggle API (for dataset downloads):
   - Download your Kaggle API credentials from https://www.kaggle.com/account
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Dependencies

- **langgraph** (0.1.17): Workflow orchestration and state management
- **langchain_anthropic** (0.32.0): Anthropic LLM integration
- **langchain_openai** (0.1.20): OpenAI LLM integration
- **langsmith** (0.1.96): Tracing and debugging
- **scikit-learn** (1.5.1): Machine learning utilities
- **kaggle** (1.6.17): Dataset downloading

## Usage

### Quick Start

1. Open the Jupyter notebook:
```bash
jupyter notebook text-minning-for-taxonomy.ipynb
```

2. Run all cells to execute the full pipeline

### Configuration Options

The system accepts the following configurable parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_case` | str | Required | Description of the taxonomy use case |
| `batch_size` | int | 200 | Number of documents per minibatch |
| `suggestion_length` | int | 30 | Max length for taxonomy suggestions |
| `cluster_name_length` | int | 10 | Max length for category names |
| `cluster_description_length` | int | 30 | Max length for category descriptions |
| `explanation_length` | int | 20 | Max length for explanations |
| `max_num_clusters` | int | 25 | Maximum number of categories |
| `max_concurrency` | int | 2 | Parallel processing limit |

### Example Usage

```python
use_case = (
    "Generate the taxonomy that can be used both to label the user intent"
    " as well as to identify any required documentation (references, how-tos, etc.)"
    " that would benefit the user."
)

stream = app.stream(
    {"documents": docs_random},
    {
        "configurable": {
            "use_case": use_case,
            "batch_size": 400,
            "max_num_clusters": 25,
        },
        "max_concurrency": 2,
    },
)

for step in stream:
    node, state = next(iter(step.items()))
    print(f"Step: {node}")
```

## How It Works

### 1. Document Summarization

Documents are processed in parallel using GPT-4o-mini to generate:
- **Summary**: Concise overview of document content
- **Explanation**: Detailed explanation of key points

The summarization uses custom prompts from LangChain Hub (`wfh/tnt-llm-summary-generation`) and XML parsing for structured output.

### 2. Minibatch Processing

Documents are split into randomized batches to:
- Ensure diversity in each processing cycle
- Enable scalable processing of large datasets
- Allow iterative refinement with different data samples

### 3. Taxonomy Generation

The initial taxonomy is created from the first minibatch using:
- Document summaries (not raw content) for efficiency
- GPT-4o for high-quality categorization
- Structured XML output with ID, name, and description

### 4. Iterative Updates

The taxonomy is refined through multiple cycles:
- Each cycle processes a different minibatch
- New categories are added as needed
- Existing categories are refined based on new data
- The system continues until all minibatches are processed

### 5. Final Review

A final review step:
- Analyzes a fresh random sample of documents
- Validates the taxonomy completeness
- Produces the final taxonomy structure

## Example Use Case: E-Commerce Reviews

The notebook demonstrates taxonomy generation using the Women's Clothing E-Commerce Reviews dataset from Kaggle:

**Input**: 1000 customer reviews
**Output**: 5 main categories:

| ID | Category | Description |
|----|----------|-------------|
| 1 | Product Fit and Return Issues | Sizing problems and returns due to incorrect fit |
| 2 | Product Quality Feedback | Compliments and concerns about quality |
| 3 | Product Features and Style Preferences | Feature discussions and recommendations |
| 4 | User Engagement Issues | Lack of user interaction or response |
| 5 | Documentation and Reference Needs | Requests for guides and how-tos |

## Project Structure

```
Text-mining-at-Scale-for-taxonomy/
├── text-minning-for-taxonomy.ipynb  # Main Jupyter notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore                        # Git ignore patterns
└── .env                              # Environment variables (not tracked)
```

## State Management

The system uses a TypedDict state structure:

```python
class TaxonomyGenerationState(TypedDict):
    documents: List[Doc]              # Documents with summaries
    minibatches: List[List[int]]      # Batch indices
    clusters: List[List[dict]]        # Taxonomy evolution trajectory
```

Each `Doc` contains:
- `id`: Unique identifier
- `content`: Original text
- `summary`: Generated summary
- `explanation`: Detailed explanation
- `category`: Assigned category (optional)

## LangSmith Integration

The project includes LangSmith tracing for:
- Debugging LLM calls
- Monitoring workflow execution
- Analyzing token usage
- Tracking performance metrics

Enable tracing by setting:
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Text-mining-for-taxonomy"
```

## Data Privacy

The `.gitignore` file excludes:
- Environment files (`.env`)
- Database files (`.db`)
- CSV datasets (`.csv`)
- Compressed archives (`.zip`)
- JSON files (`.json`)

Ensure sensitive data and API keys are never committed to version control.

## Performance Considerations

- **Parallel Processing**: Document summarization is batched with configurable concurrency
- **Context Efficiency**: Summaries are used instead of raw content to reduce token usage
- **Caching**: LangChain InMemoryCache reduces redundant LLM calls
- **Batch Size**: Adjust based on your dataset size and memory constraints

## Limitations

- Requires OpenAI API access (GPT-4o and GPT-4o-mini)
- Processing time scales with dataset size
- LLM costs increase with document volume
- Best suited for English language content (can be adapted)

## Future Enhancements

Potential improvements:
- Multi-language support
- Custom LLM provider options
- Automatic batch size optimization
- Interactive taxonomy refinement
- Export to various formats (JSON, CSV, etc.)
- Integration with classification models

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is available for educational and research purposes.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Uses OpenAI's GPT-4o model
- Prompts adapted from LangChain Hub
- Dataset from [Kaggle](https://www.kaggle.com/)

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: This project requires API keys and may incur costs based on LLM usage. Monitor your usage to control expenses.
