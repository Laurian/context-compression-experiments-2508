#!/usr/bin/env python3
"""
DSPy GEPA optimizer for context compression prompts.

This script loads observation data, extracts context and query pairs,
and uses DSPy GEPA to optimize the contextual compression prompt.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
from contextlib import contextmanager

import dspy
import weave
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

# Suppress wandb warnings
warnings.filterwarnings("ignore", category=UserWarning, module="wandb")
os.environ["WANDB_SILENT"] = "true"

# Create custom filter for parallelizer errors
class ParallelizerErrorFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == 'dspy.utils.parallelizer' and record.levelno == logging.ERROR)

# Add the filter to root logger
logging.getLogger().addFilter(ParallelizerErrorFilter())

# Suppress verbose DSPy output
os.environ["DSP_VERBOSE"] = "false"
logging.getLogger("dspy").setLevel(logging.WARNING)
logging.getLogger("dspy.teleprompt").setLevel(logging.WARNING)
logging.getLogger("dspy.teleprompt.gepa").setLevel(logging.INFO)
logging.getLogger("dspy.utils.parallelizer").setLevel(logging.CRITICAL)  # Suppress parallelizer errors


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout during optimization."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class DataLoader:
    """Load and parse observation data to extract context/query pairs."""
    
    def __init__(self, observations_dir: str, gpt4o_dir: str):
        self.observations_dir = Path(observations_dir)
        self.gpt4o_dir = Path(gpt4o_dir)
    
    def extract_context_and_query(self, input_messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """Extract context and query from input messages."""
        context = None
        query = None
        
        for message in input_messages:
            content = message.get("content", "")
            
            # Extract context from <context>...</context> tags (usually in system message)
            if not context:
                context_match = re.search(r'<context>(.*?)</context>', content, re.DOTALL)
                if context_match:
                    context = context_match.group(1).strip()
            
            # Extract query from <query>...</query> tags (usually in user message)  
            if not query:
                query_match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
                if query_match:
                    query = query_match.group(1).strip()
        
        return context, query
    
    def load_observations(self) -> List[Dict]:
        """Load all observation files and extract relevant data."""
        observations = []
        
        # Get all JSON files except those starting with "_"
        observation_files = [f for f in self.observations_dir.glob("*.json") if not f.name.startswith("_")]
        logger.info(f"Loading {len(observation_files)} observation files...")
        
        for obs_file in tqdm(observation_files, desc="Loading observations"):
            try:
                with open(obs_file, 'r') as f:
                    data = json.load(f)
                
                # Extract context and query from input
                context, query = self.extract_context_and_query(data.get("input", []))
                
                # Skip observations that are too large to process efficiently
                if context and len(context) > 25000:  # Limit context to 25k chars for stability
                    context = context[:25000] + "... [truncated]"
                    logger.debug(f"Truncated context for observation {data.get('id')} from {len(context)} to 25k chars")
                
                if context and query:
                    obs_data = {
                        "id": data.get("id"),
                        "context": context,
                        "query": query,
                        "original_output": data.get("output", {}).get("content", ""),
                    }
                    
                    # Load corresponding GPT-4o success if available
                    gpt4o_file = self.gpt4o_dir / f"{data.get('id')}.json"
                    if gpt4o_file.exists():
                        with open(gpt4o_file, 'r') as f:
                            gpt4o_data = json.load(f)
                            obs_data["target_output"] = gpt4o_data.get("output", "")
                            obs_data["has_target"] = True
                    else:
                        obs_data["has_target"] = False
                    
                    observations.append(obs_data)
                    
            except Exception as e:
                logger.warning(f"Error processing {obs_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(observations)} valid observations")
        logger.info(f"Observations with GPT-4o targets: {sum(1 for obs in observations if obs['has_target'])}")
        
        return observations


# Define the base prompt from README.md
BASE_COMPRESSION_PROMPT = """You are tasked with performing a contextual compression of a document as part of a system that processes multiple documents. Your goal is to extract only the essential parts of the given context that are relevant to a specific query.
This process helps in focusing on the most important information and reducing noise in the context.
The query might refer to multiple documents, consider how does apply to a single document in the context as multiple documents might be relevant.

Your task is to extract any parts of the context that are directly relevant to answering this question. Follow these guidelines:

1. Only extract text *AS IS* that is directly related to the query.
2. Do not modify, paraphrase, or summarize the extracted text. Copy it exactly as it appears in the context.
3. You may extract multiple separate parts if necessary.
4. If a header relates to the query, extract also the text under that section.
5. Preserve headings and subheadings when extracting.
6. If you find no relevant information in the context, output "NO_OUTPUT"."""


class ContextCompressor(dspy.Module):
    """DSPy module for contextual compression starting with README prompt."""
    
    def __init__(self):
        super().__init__()
        # Use simple signature like working example
        self.step = dspy.Predict("user -> reply")
        self.step.signature.instructions = BASE_COMPRESSION_PROMPT
        
        logger.info(f"Initialized ContextCompressor with base prompt: {BASE_COMPRESSION_PROMPT[:100]}...")
    
    def forward(self, user: str) -> dspy.Prediction:
        return self.step(user=user)


def evaluate_compression(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Enhanced metric function for GEPA optimization following working example pattern.
    Returns dspy.Prediction with score and feedback for GEPA's reflection mechanism.
    """
    # Extract prediction output 
    pred_output = pred.reply.strip() if hasattr(pred, 'reply') else str(pred).strip()
    
    # Get gold standard information
    has_target = gold.get("has_target", False)
    target_output = gold.get("target_output", "")
    user_input = gold.get("user", "")
    query = gold.get("query", "")
    
    # Simple scoring logic
    if has_target:
        if pred_output == "NO_OUTPUT" and target_output != "NO_OUTPUT":
            score = 0.0
            feedback = "Model incorrectly output NO_OUTPUT when relevant content exists"
        elif pred_output == "NO_OUTPUT" and target_output == "NO_OUTPUT":
            score = 1.0
            feedback = "Model correctly identified no relevant content"
        elif target_output != "NO_OUTPUT" and pred_output != "NO_OUTPUT":
            # Simple length-based similarity as proxy
            target_len = len(target_output.strip())
            pred_len = len(pred_output.strip())
            if target_len > 0:
                length_ratio = min(pred_len, target_len) / max(pred_len, target_len)
                score = length_ratio * 0.8  # Max score 0.8 for having content
                feedback = f"Content extracted, length similarity: {length_ratio:.2f}"
            else:
                score = 0.0
                feedback = "Target has no content but prediction does"
        else:
            score = 0.0
            feedback = "Prediction-target mismatch"
    else:
        score = 0.0
        feedback = "No target available for evaluation"
    
    return dspy.Prediction(score=score, feedback=feedback)


class DSPyGEPAOptimizer:
    """Main optimizer class using DSPy GEPA."""
    
    def __init__(self, observations: List[Dict]):
        self.observations = observations
        self.setup_dspy()
        self.setup_weave()
    
    def setup_dspy(self):
        """Configure DSPy with OpenAI model."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Configure custom cache directory (use .env value if set, otherwise default)
        cache_dir_env = os.getenv('DSPY_CACHEDIR')
        if cache_dir_env:
            cache_dir = Path(cache_dir_env)
        else:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['DSPY_CACHEDIR'] = str(cache_dir)
        
        # Use GPT-4o-mini as the target model we want to optimize for
        lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key)
        dspy.configure(lm=lm)
        logger.info(f"DSPy configured with GPT-4o-mini, cache dir: {cache_dir}")
        
        # Cache is configured above - can be manually cleared by deleting cache directory
    
    def setup_weave(self):
        """Initialize W&B Weave for experiment tracking."""
        wandb_project = os.getenv("WANDB_PROJECT", "context-compression-experiments")
        wandb_api_key = os.getenv("WANDB_API_KEY")
        
        try:
            if wandb_api_key and not wandb_api_key.startswith("#"):  # Skip if commented out
                wandb.login(key=wandb_api_key, relogin=True)
                weave.init(wandb_project)
                logger.info(f"Weave initialized for project: {wandb_project}")
                return True
            else:
                logger.info("WANDB_API_KEY not found or commented out, skipping Weave initialization")
                return False
        except Exception as e:
            logger.warning(f"Failed to initialize Weave: {e}. Continuing without tracking.")
            return False
    
    def prepare_dataset(self, max_examples: Optional[int] = None) -> List[dspy.Example]:
        """Convert observations to DSPy examples."""
        examples = []
        
        # Prioritize examples with targets for training
        observations_with_targets = [obs for obs in self.observations if obs.get("has_target", False)]
        observations_without_targets = [obs for obs in self.observations if not obs.get("has_target", False)]
        
        # Use observations with targets first, then fill with others
        selected_obs = observations_with_targets
        if max_examples and len(selected_obs) < max_examples:
            remaining = max_examples - len(selected_obs)
            selected_obs.extend(observations_without_targets[:remaining])
        elif max_examples:
            selected_obs = selected_obs[:max_examples]
        
        for obs in selected_obs:
            # Create user message template like working example
            user_message = f"""Here is the context document:

<context>
{obs["context"]}
</context>

Now, consider the following query:

<query>
{obs["query"]}
</query>

Now, proceed with the task using the provided context and query."""

            example = dspy.Example(
                user=user_message,
                context=obs["context"],
                query=obs["query"],
                has_target=obs["has_target"],
                target_output=obs.get("target_output", ""),
                expected_output=obs.get("target_output", ""),
                original_output=obs["original_output"]
            ).with_inputs("user")
            
            examples.append(example)
        
        logger.info(f"Prepared {len(examples)} examples for optimization")
        return examples
    
    @weave.op()
    def run_optimization(
        self, 
        max_examples: int = 100,
        population_size: int = 10,
        generations: int = 5,
        train_split: float = 0.8
    ) -> Dict:
        """Run GEPA optimization."""
        
        # Prepare dataset
        examples = self.prepare_dataset(max_examples)
        
        # Split into train/validation
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        logger.info(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")
        
        # Initialize model
        model = ContextCompressor()
        
        # Set up GEPA optimizer
        from dspy.teleprompt import GEPA
        
        # Create a reflection LM for GEPA (using GPT-4o for reflection)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        reflection_lm = dspy.LM('openai/gpt-4o', api_key=openai_api_key, temperature=1.0, max_tokens=8000)
        
        # Check if W&B is available
        use_wandb = bool(os.getenv("WANDB_API_KEY") and not os.getenv("WANDB_API_KEY").startswith("#"))
        
        teleprompter = GEPA(
            metric=evaluate_compression,
            auto="medium",  # Use medium automation like working example
            num_threads=4,  # Limit threads for rate limiting
            track_stats=True,
            reflection_lm=reflection_lm
        )
        
        logger.info(f"GEPA initialized with W&B tracking: {use_wandb}")
        
        # Custom callback to log only prompt changes
        def prompt_callback(iteration, prompt):
            logger.info(f"Iteration {iteration} - New prompt: {prompt}")
        
        # Run optimization
        logger.info("Starting GEPA optimization...")
        logger.info("Suppressing verbose context/query output - showing only key progress...")
        
        # Suppress verbose stdout during optimization while preserving important logs
        with suppress_stdout():
            optimized_model = teleprompter.compile(
                model, 
                trainset=train_examples,
                valset=val_examples
            )
        
        logger.info("GEPA optimization completed!")
        
        # Evaluate final model
        correct = 0
        total = len(val_examples)
        
        for example in val_examples:
            pred = optimized_model(user=example.user)
            eval_result = evaluate_compression(example, pred, None, "compression", None)
            score = eval_result.score if hasattr(eval_result, 'score') else eval_result
            correct += score
        
        accuracy = correct / total if total > 0 else 0
        
        results = {
            "accuracy": accuracy,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "population_size": population_size,
            "generations": generations,
        }
        
        logger.info(f"Final validation accuracy: {accuracy:.3f}")
        
        return {
            "model": optimized_model,
            "results": results,
            "examples": examples
        }


def main():
    """Main execution function."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    observations_dir = project_root / "data" / "observations"
    gpt4o_dir = project_root / "data" / "gpt-4o"
    
    # Verify data directories exist
    if not observations_dir.exists():
        raise FileNotFoundError(f"Observations directory not found: {observations_dir}")
    if not gpt4o_dir.exists():
        raise FileNotFoundError(f"GPT-4o directory not found: {gpt4o_dir}")
    
    # Load data
    logger.info("Loading observation data...")
    loader = DataLoader(str(observations_dir), str(gpt4o_dir))
    observations = loader.load_observations()
    
    if not observations:
        raise ValueError("No valid observations found")
    
    # Run optimization
    optimizer = DSPyGEPAOptimizer(observations)
    
    # Start with a smaller subset for initial testing
    optimization_results = optimizer.run_optimization(
        max_examples=50,  # Start small
        population_size=8,
        generations=3,
        train_split=0.8
    )
    
    # Save results with timestamp and specific naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = project_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific subdirectory
    experiment_name = f"gepa_context_compression_{timestamp}"
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimized model with proper file extension
    model_file = experiment_dir / "optimized_model.json"
    optimized_model = optimization_results["model"]
    optimized_model.save(str(model_file))
    
    # Save detailed results with experiment parameters
    results_with_metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "script": "dspy_gepa_optimizer.py",
        "optimization_type": "GEPA",
        "model_target": "gpt-4o-mini",
        "parameters": {
            "max_examples": 50,
            "population_size": 8, 
            "generations": 3,
            "train_split": 0.8
        },
        "results": optimization_results["results"]
    }
    
    with open(experiment_dir / "experiment_results.json", "w") as f:
        json.dump(results_with_metadata, f, indent=2)
    
    # Also save a summary in the main results directory
    with open(results_dir / f"{experiment_name}_summary.json", "w") as f:
        summary = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "accuracy": optimization_results["results"]["accuracy"],
            "script": "dspy_gepa_optimizer.py"
        }
        json.dump(summary, f, indent=2)
    
    logger.info(f"Optimization complete! Results saved to {experiment_dir}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Final accuracy: {optimization_results['results']['accuracy']:.3f}")


if __name__ == "__main__":
    main()