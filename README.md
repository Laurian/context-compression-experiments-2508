# Context Compression Prompt Experiments

I have in production an Open Deep Research-like Agentic RAG over an internal knowledge base. Upon retrieval each document needs context compression: extracting the content that is relevant to the query that retrieved the document, extracting paragraphs by semantic similarity is not good enough and a LLM is way more accurate. 

The prompt (inspired from the LangChain's [LLMChainExtractor](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.document_compressors.chain_extract.LLMChainExtractor.html) [prompt](https://github.com/langchain-ai/langchain/blob/b999f356e86b706d943ac2c3ee9b21a0cffeefa5/libs/langchain/langchain/retrievers/document_compressors/chain_extract_prompt.py)) I use is:

```md
You are tasked with performing a contextual compression of a document as part of a system that processes multiple documents. Your goal is to extract only the essential parts of the given context that are relevant to a specific query.
This process helps in focusing on the most important information and reducing noise in the context.
The query might refer to multiple documents, consider how does apply to a single document in the context as multiple documents might be relevant.

Your task is to extract any parts of the context that are directly relevant to answering this question. Follow these guidelines:

1. Only extract text *AS IS* that is directly related to the query.
2. Do not modify, paraphrase, or summarize the extracted text. Copy it exactly as it appears in the context.
3. You may extract multiple separate parts if necessary.
4. If a header relates to the query, extract also the text under that section.
5. Preserve headings and subheadings when extracting.
6. If you find no relevant information in the context, output "NO_OUTPUT".

Here is the context document:

<context>[MARKDOWN_CONTENT]</context>

Now, consider the following query:

<query>[EXAMPLE_QUERY]</query>

Now, proceed with the task using the provided context and query.
```

Which performs well with `gpt-4o` and not that great with `gpt-4o-mini` and in production we frequently fallback to `gpt-4o-mini` due to rate limits on Azure OpenAI. 

I filtered out 1000+ `gpt-4o-mini` traces from LangFuse that failed contextual compressions (nothing extracted from the document: `NO_OUTPUT`) and re-ran them against `gpt-4o` which was successful in about 296 of them. That's my sample dataset of *document × query* pairs for which I would like to make `gpt-4o-mini` perform better.

*Please note: Certain domain-specific details have been intentionally omitted for privacy and confidentiality.*


## DSPy GEPA (Genetic Pareto)

I first used [DSPy](https://dspy.ai/api/optimizers/GEPA/) [GEPA (Genetic-Pareto)](https://github.com/gepa-ai/gepa) to optimise the prompt, basically had Claude Code understand the input dataset and after a couple of iterations on some errors and misunderstandings of the goal, I got it running.


Running [dspy_gepa_optimizer.py](scripts/dspy_gepa_optimizer.py):

```
Running DSPy GEPA optimization...
This will optimize the context compression prompt using genetic algorithms.
Make sure you have configured your .env file with API keys.
uv run python scripts/dspy_gepa_optimizer.py
INFO:__main__:Loading observation data...
INFO:__main__:Loaded 296 valid observations
INFO:__main__:Observations with GPT-4o targets: 296
INFO:__main__:DSPy configured with GPT-4o-mini, cache dir: cache/dspy
INFO:__main__:WANDB_API_KEY not found or commented out, skipping Weave initialization
WARNING:weave.trace.op:Warning: Traces will not be logged. Call weave.init to log your traces to a project.
 (subsequent messages of this type will be suppressed)
INFO:__main__:Prepared 50 examples for optimization
INFO:__main__:Training on 40 examples, validating on 10
INFO:__main__:Initialized ContextCompressor with base prompt: You are tasked with performing a contextual compression of a document as part of a system that proce...
INFO:__main__:GEPA initialized with W&B tracking: False
INFO:__main__:Starting GEPA optimization...
INFO:__main__:Suppressing verbose context/query output - showing only key progress...
2025/08/30 19:44:43 INFO dspy.teleprompt.gepa.gepa: Running GEPA for approx 740 metric calls of the program. This amounts to 14.80 full evals on the train+val set.
2025/08/30 19:44:43 INFO dspy.teleprompt.gepa.gepa: Using 10 examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget.
2025/08/30 19:45:09 INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.3226210710799752
2025/08/30 19:45:09 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Selected program 0 score: 0.3226210710799752
2025/08/30 19:45:33 INFO dspy.teleprompt.gepa.gepa: Iteration 1:

Proposed new text for step: You are tasked with performing a contextual compression of a document as part of a system that processes multiple documents. Your goal is to extract only the essential parts of the given context that are relevant to a specific query. This process helps in focusing on the most important information and reducing noise in the context. The query might refer to multiple documents, consider how this applies to a single document in the context as multiple documents might be relevant.

Your task is to extract any parts of the context that are directly relevant to answering this question. Ensure you follow these guidelines to fulfill the task accurately:

1. Only extract text *AS IS* that is directly related to the query.
2. Do not modify, paraphrase, or summarize the extracted text. Copy it exactly as it appears in the context.
3. You may extract multiple separate parts if necessary.
4. If a header relates to the query, extract also the text under that section.
5. Preserve headings and subheadings when extracting.
6. If you find no relevant information in the context, output "NO_OUTPUT".

Specific Details and Strategies:
- Focus on identifying sections or pieces of text that explicitly address the keywords or topics mentioned in the query.
- Double-check sections that discuss plans, strategies, innovative approaches, or solutions, especially when the query is about strategies or innovative steps. These terms often appear in problem statements, solution overviews, or specific projects and initiatives.
- Consider the overall aim of the document's project or initiative, as this often contains relevant strategies and execution plans, which may directly answer the query.
- Always ensure complete fidelity to the original text in the document; do not alter or modify it in any way.

These steps and considerations ensure that the information extracted is both relevant and authentic to the source material, providing a precise answer to the posed query.
```

It is fascinating to watch the variants it evolves:

```
2025/08/30 20:23:22 INFO dspy.teleprompt.gepa.gepa: Iteration 42: Selected program 11 score: 0.4210539502301618
2025/08/30 20:23:42 INFO dspy.teleprompt.gepa.gepa: Iteration 42:

Proposed new text for step: Your task is to extract relevant sections from a provided context document in response to a specific query, maintaining the integrity and structure of the source material. Follow these detailed guidelines to perform the task:

1. **Exact Text Extraction**: Extract sections of text exactly as they appear in the source document. Do not alter, paraphrase, or reformat the text. Maintain original headings, subheadings, bullet points, and any other structural elements present in the context.

2. **Comprehensive Coverage**: When relevant content spans multiple sections or paragraphs across the document, ensure all pertinent parts are extracted. The goal is to fully address varying facets of the query.

3. **Structure Preservation**: Maintain the original structural organization of the extracted sections. Include section headings with associated content if they provide context relevant to the query.

4. **Relevance Assessment**: Focus on extracting sections that are directly relevant to the query. Prioritize content detailing strategies, solution frameworks, innovative methodologies, and challenges or barriers pertinent to the query.

5. **Precision in Domain-Specific Focus**:
   - For ████████ management-related queries, emphasize information on efficient █████████ technologies, ███████████ techniques, and ███████████████ management practices.
   - For █████████ improvement-focused queries, focus on innovative ██████████ practices, █████████████ management, and ██████████ solutions.
   - For ██████████-related queries, highlight ██████████ challenges, policies, and solutions related to █████████, especially addressing █████████████ impacts.

6. **Handling Unrelated Content**: If the document lacks pertinent information relevant to the posed query, provide a response indicating "NO_OUTPUT".

7. **Applying Domain Knowledge**: Use specialized knowledge in relevant domains to assess the relevance of the content. Understand key concepts and terminology in areas such as ███████████, █████████████ technology, ████████████ innovations, and regulatory challenges.

8. **Analyzing Barriers and Solutions**: Highlight documented challenges and corresponding solutions. This analysis is crucial for understanding systemic approaches and implementations within the project or title.

9. **Verification of Contextual Integrity**: Ensure the extracted information accurately represents the intended meaning and specifics from the source material, without introducing extraneous interpretation or context loss.

By adhering to these guidelines, ensure that your responses are accurate, relevant, and comprehensive in addressing the posed queries while remaining faithful to the source context.

2025/08/30 20:23:51 INFO dspy.teleprompt.gepa.gepa: Iteration 42: New subsample score is not better, skipping
```

After 1h and 75 iterations, the final result is:

```
2025/08/30 20:49:13 INFO dspy.teleprompt.gepa.gepa: Iteration 75: Selected program 4 score: 0.329291347392866
2025/08/30 20:49:22 INFO dspy.teleprompt.gepa.gepa: Iteration 75: Proposed new text for step:

You are tasked with extracting relevant information from a document based on a specific query regarding strategies and projects. The aim is to identify and extract verbatim sections from the document that directly relate to answering the query, focusing solely on relevant parts without paraphrasing or summarizing.

To perform this task accurately, follow these guidelines:

1. **Direct Extraction**: Extract text exactly as found in the document. The text must directly answer or relate to the query. Ensure no modification, paraphrasing, or summarizing alters the original text.

2. **Contextual Relevance**: Identify sections that explicitly discuss strategies, methods, innovative approaches, or plans that relate to the query, especially if they pertain to geographical locations or specific themes such as ██████████, █████████ for █████████, or ██████████ management in ████████.

3. **Multiple Sections**: If necessary, extract multiple sections to comprehensively capture the relevant information for the query. Ensure all parts extracted are directly tied to answering the query.

4. **Preserve Structure**: If headings or subheadings accompany relevant information, extract them along with the subsequent text. Maintaining this structure helps preserve the context in which the information appears.

5. **No Relevant Information**: If the given context does not provide any relevant information to the query, return "NO_OUTPUT".

6. **Domain Specific Knowledge**: Recognize key focuses within the document relevant to topics such as strategies for reducing ███████████, █████████ in ███████████, or technical methods for ███████████. Look for plans, outcomes, problem statements, and innovative solutions that align with the specified areas.

7. **Focus on Detail**: Pay attention to specific details that denote initiatives, methods, and strategic plans within the context of the query. These could be specific projects, phases, timelines, or evidence of effectiveness that are directly related to the query topic.

8. **Precision and Accuracy**: Ensure the extracted information is an accurate representation of the original text, and verify that it pertains directly to the core of the query, leveraging any specific examples or documented evidence within the context.

By adhering to these instructions, you will ensure that the extracted content remains true to its source and provides a precise and accurate answer to the query.
2025/08/30 20:49:31 INFO dspy.teleprompt.gepa.gepa: Iteration 75: New subsample score is not better, skipping
INFO:__main__:GEPA optimization completed!
INFO:__main__:Final validation accuracy: 0.447
INFO:__main__:Optimization complete! Results saved to /Users/laurian/Projects/G/context-compression-experiments-2508/data/results/gepa_context_compression_20250830_204931
INFO:__main__:Experiment: gepa_context_compression_20250830_204931
INFO:__main__:Final accuracy: 0.447
```

Running this prompt on the 296 documents with `gpt-4o-mini` extracted content in 62% of cases (as opposed to the original 0%).


## TextGrad

I ran the same initial prompt with [TextGrad](https://github.com/zou-group/textgrad) (again script generated with Claude Code and some minor corrections done with Cursor), and running [textgrad_optimizer.py](scripts/textgrad_optimizer.py) I got this prompt:

```
You are tasked with performing a contextual compression of a document as part of a system that processes multiple documents. Your goal is to extract only the essential parts of the given context that are relevant to a specific query. This process helps in focusing on the most important information and reducing noise in the context. The query might refer to multiple documents, consider how this applies to a single document in the context as multiple documents might be relevant.

Your task is to extract any parts of the context that are directly relevant to answering this question. Follow these guidelines:

1. Broaden your scope to include a variety of geographic and thematic contexts as required by the query, such as ████████████ initiatives across all of █████████, not just a specific region or project.
2. Prioritize content that is directly related to the specific geographic and thematic context of the query, such as ████████-specific data, policies, and challenges when the query pertains to ████████████ in the ██████████.
3. Deprioritize or exclude global statistics unless they have a direct and significant impact on the context relevant to the query.
4. Extract information from diverse sources within the context, ensuring a comprehensive overview by including multiple perspectives and projects from different organizations and government bodies.
5. Only extract text *AS IS* that is directly related to the query, maintaining original phrasing and key terms, such as "███████ █████████" and "█████████ ███████," to preserve the original intent and emphasis.
6. You may extract multiple separate parts if necessary.
7. If a header relates to the query, extract also the text under that section, ensuring all relevant sections like "Innovation" and "Subject Area" are included.
8. Preserve headings and subheadings when extracting, and use bullet points to organize content clearly, maintaining the original structure and format.
9. Critically assess the relevance of each section before extraction, ensuring that only the most pertinent information is included.
10. Emphasize the importance of identifying technological advances and case studies within the context, especially if they align with the query's requirements, such as ███████ ████████ approaches in █████████.
11. Review the "██████████ ████████" section to ensure comprehensive coverage of relevant solutions and their impacts.
12. Highlight key government and NGO initiatives that have a significant impact, ensuring they are emphasized in the extracted content.
13. If you find no relevant information in the context, output "NO_OUTPUT". Ensure this decision is made after a thorough review of the context.
14. Consider the context as a whole to understand the relationships between different pieces of information and their relevance to the query.
15. Reinforce the importance of maintaining a clear hierarchy of information in the extracted content, ensuring critical information is easily accessible.
16. Extract specific data or statistics related to ███████████ challenges and opportunities in █████████, citing recent studies or government reports where applicable.
17. Include information about government policies, initiatives, or collaborations with organizations that are specifically aimed at addressing these issues in █████████.
18. Conduct a thorough secondary review to ensure no relevant information was missed, critically assessing the extracted content against the original text to ensure it provides a balanced and comprehensive view of the topic.
```

Which extracts with `gpt-4o-mini` from 79% of the 296 documents.

# Hybrid: DSPy GEPA + TextGrad

Also running [TextGrad](https://github.com/zou-group/textgrad) on the [DSPy GEPA](https://dspy.ai/api/optimizers/GEPA/) optimised prompt yields an even smaller prompt:

```
You are tasked with performing a contextual compression of a document to extract only the essential parts relevant to a specific query. This involves sifting through a given context document, which could cover various domains such as town development, water conservation, or housing justice, and identifying pertinent sections or text that directly address the topics or keywords mentioned in the query.

To execute this task:

1. **Relevance Enhancement**: Prioritize content that directly addresses the specific challenges and solutions related to ██████ ███████ projects. Focus on identifying and extracting information that discusses the historical context, policy implications, and specific community impacts of ███████ and ██████████████.

2. **Completeness Improvement**: Extract detailed examples or case studies that illustrate the challenges and solutions. Look for specific data points, such as statistics on ███████████ or success stories from initiatives, to provide a comprehensive view of the issues.

3. **Exactness Reinforcement**: Maintain the exact wording from the source material, especially for critical data or statements. Prioritize verbatim extraction over paraphrasing to preserve the original meaning and nuance of the information.

4. **Format Preservation**: Use consistent heading levels and ensure clear separation between different sections to maintain the original format of the source material. This will enhance readability and help maintain the original structure.

5. **Instruction Clarity**: Focus on extracting content that aligns with the evaluation criteria, emphasizing relevance, completeness, exactness, and format preservation. Clear and specific instructions will guide the model to produce outputs that better meet the objective function.

6. **Feedback Loop Integration**: Implement a feedback loop mechanism to allow for continuous improvement. Log instances where the model's output lacks specificity or completeness and analyze them to refine the model's decision-making process.

These guidelines are designed to ensure the information extracted is both relevant and authentic to the source material, providing an accurate and focused response to the query.
```

Which successfully extracts with `gpt-4o-mini` content from all of the documents (100% success rate).

Extracting something is better than nothing, but I wanted to have a way to compare the "coverage" of extractions, this time I used OpenCode (with Gemini 2.5 Pro Preview via OpenRouter) to visualise all the documents and the extracted lines and their overlap (using sentence-level semantic similarity as the extraction is not always verbatim).

Here is a map of 296 documents as grey horizontal lines, where on each document the vertical lines represent a matched text line (a long paragraph counts a single line) where blue are lines matched only by `gpt-4o` with original prompt, red are lines matched with `gpt-4o-mini` with the best prompt, and white are matched by both (agreement).

![Coverage Map](coverage_map.png)

I probably need to refine how I do sentence/line matching to be sure I visualise the coverage without false positives (repeated sentences in the document might skew some results), but this is a good start in fixing the context compression issues I had in production.

## How To Run Optimization Experiments

This project provides a comprehensive optimization pipeline with three different approaches to improve context compression prompts for GPT-4o-mini. Here's how to run each optimization method and test the results.

### Prerequisites

1. **Environment Setup**:
   ```bash
   make setup  # Create virtual environment and install dependencies
   ```

2. **Configure API Keys**:
   Copy `.env.template` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.template .env
   # Edit .env and add: OPENAI_API_KEY=your_api_key_here
   ```

3. **Verify Data Structure**:
   ```bash
   make data-check  # Verify observation and GPT-4o data directories exist
   ```

### Optimization Methods

#### 1. DSPy GEPA Optimization (Primary Method)

Run genetic algorithm-based prompt optimization:

```bash
make optimize
```

This will:
- Load ~300 observation files from `data/observations/`
- Use ~300 GPT-4o success cases from `data/gpt-4o/` as targets
- Run GEPA genetic optimization for ~75 iterations (1+ hours)
- Save results to `data/results/gepa_context_compression_{timestamp}/`
- Generate an optimized DSPy model and experiment metadata

**Expected Output**: Improved validation accuracy from ~32% to ~45% (as shown in example above)

#### 2. TextGrad Optimization (Alternative Method)

Run textual gradient descent optimization from the base prompt:

```bash
make optimize-textgrad
```

This will:
- Start from the original context compression prompt
- Use TextGrad's textual gradient descent over 8 iterations
- Apply natural language feedback for prompt refinement
- Save results to `data/results/textgrad_context_compression_{timestamp}/`
- Generate optimized prompt text and experiment results

**Expected Behavior**: Different optimization approach using text-based gradients

#### 3. Hybrid TextGrad+GEPA Optimization (Advanced Method)

Run TextGrad optimization starting from the latest GEPA-optimized prompt:

```bash
# First ensure you have a GEPA baseline
make optimize

# Then refine it with TextGrad
make optimize-textgrad-gepa
```

This will:
- Automatically find the latest GEPA optimization result
- Use the GEPA-optimized prompt as starting point for TextGrad
- Apply textual gradient descent refinement over 8 iterations  
- Save results to `data/results/textgrad_gepa_context_compression_{timestamp}/`
- Track improvements over the GEPA baseline

**Expected Behavior**: Potential further improvement over GEPA-only results

### Testing Optimized Prompts

After running optimization, test the results against all observations:

#### Test GEPA Results
```bash
make test-gepa
```
- Tests latest GEPA-optimized prompt against all 1,700+ observations
- Uses GPT-4o-mini to evaluate performance
- Saves results to `data/tests/gpt-4o-mini-test-{timestamp}/`

#### Test TextGrad Results
```bash
make test-textgrad
```
- Tests latest TextGrad-optimized prompt
- Saves results to `data/tests/textgrad-gpt-4o-mini-test-{timestamp}/`

#### Test Hybrid Results
```bash
make test-textgrad-gepa
```
- Tests latest TextGrad+GEPA hybrid prompt
- Saves results to `data/tests/textgrad-gepa-gpt-4o-mini-test-{timestamp}/`

### Complete Experimental Pipeline

For a comprehensive comparison of all methods:

```bash
# 1. Run all optimization approaches
make optimize                    # GEPA baseline (~1 hour)
make optimize-textgrad          # TextGrad from scratch (~30 mins)
make optimize-textgrad-gepa     # TextGrad refining GEPA (~30 mins)

# 2. Test all optimized prompts
make test-gepa                  # Test GEPA results (~45 mins for 1700 docs)
make test-textgrad             # Test TextGrad results (~45 mins)
make test-textgrad-gepa        # Test hybrid results (~45 mins)

# 3. Compare results
# Check success rates in the respective data/tests/ directories
```

### Process Management

Long-running optimizations can be killed if needed:

```bash
make kill-optimizer         # Kill DSPy GEPA processes
make kill-textgrad         # Kill TextGrad processes  
make kill-textgrad-gepa    # Kill TextGrad+GEPA processes

# Kill testing processes
make kill-test-gepa
make kill-test-textgrad
make kill-test-textgrad-gepa
```

### Understanding Results

#### Optimization Results Structure
- **GEPA**: `data/results/gepa_context_compression_{timestamp}/`
  - `optimized_model.json` - DSPy model with optimized prompt
  - `experiment_results.json` - Detailed optimization metadata
  - `{experiment_name}_summary.json` - Quick summary with accuracy

- **TextGrad**: `data/results/textgrad_context_compression_{timestamp}/`  
  - `optimized_prompt.txt` - Final optimized prompt text
  - `experiment_results.json` - Optimization history and results

- **Hybrid**: `data/results/textgrad_gepa_context_compression_{timestamp}/`
  - Contains both GEPA source tracking and TextGrad refinements

#### Testing Results Structure
Each test creates detailed results in `data/tests/`:
- `test_summary.json` - Overall success rate and token usage
- `test_config.json` - Test configuration and prompt used
- `results/{observation_id}.json` - Individual test results per document

#### Performance Expectations
Based on the example results:
- **Original prompt**: 0% success rate with GPT-4o-mini
- **GEPA optimized**: ~45% validation accuracy, ~62% test success rate  
- **TextGrad optimized**: ~79% test success rate
- **Hybrid optimized**: Up to 100% success rate (extracting something vs nothing)

### Troubleshooting

1. **Missing API Key**: Ensure `OPENAI_API_KEY` is set in `.env`
2. **Rate Limits**: Optimizations use rate limiting, but may still hit limits
3. **Memory Issues**: Large contexts are truncated to 25k characters automatically
4. **Long Runtimes**: GEPA optimization can take 1+ hours for full dataset

### Additional Commands

```bash
make help           # Show all available commands
make clear-cache    # Clear DSPy cache if needed
make data-check     # Verify data structure
make notebook       # Start Jupyter Lab for analysis
```

This pipeline enables systematic comparison of different optimization approaches to find the best method for improving GPT-4o-mini's context compression performance.
