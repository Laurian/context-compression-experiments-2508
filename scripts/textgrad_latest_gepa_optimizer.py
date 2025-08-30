#!/usr/bin/env python3
"""
TextGrad optimizer for context compression prompts starting from latest GEPA prompt.

This script loads the latest GEPA-optimized prompt as the starting point,
then uses TextGrad to further optimize it. This allows for iterative optimization
where GEPA provides a good initial prompt and TextGrad refines it further.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import numpy as np
from datetime import datetime
import time

import textgrad as tg
import weave
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

# Suppress wandb warnings
warnings.filterwarnings("ignore", category=UserWarning, module="wandb")
os.environ["WANDB_SILENT"] = "true"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_gepa_result() -> Optional[Path]:
    """Find the most recent GEPA optimization result directory."""
    results_dir = Path(__file__).parent.parent / "data" / "results"
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return None
    
    # Find all GEPA result directories
    gepa_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("gepa_context_compression_")]
    
    if not gepa_dirs:
        logger.error("No GEPA optimization results found")
        return None
    
    # Sort by creation time and get the latest
    latest_dir = max(gepa_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Found latest GEPA result: {latest_dir}")
    return latest_dir


def load_gepa_optimized_prompt(gepa_dir: Path) -> Optional[str]:
    """Load the optimized prompt from the GEPA result."""
    model_file = gepa_dir / "optimized_model.json"
    
    if not model_file.exists():
        logger.error(f"Optimized model file not found: {model_file}")
        return None
    
    try:
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        
        # Extract the optimized instructions from the model
        step_data = model_data.get("step", {})
        signature = step_data.get("signature", {})
        instructions = signature.get("instructions", "")
        
        if not instructions:
            logger.error("No instructions found in optimized model")
            return None
        
        logger.info(f"Loaded GEPA-optimized prompt ({len(instructions)} chars)")
        logger.info(f"Prompt preview: {instructions[:200]}...")
        return instructions
        
    except Exception as e:
        logger.error(f"Error loading optimized model: {e}")
        return None


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


class ContextualCompressionModel:
    """TextGrad model wrapper for contextual compression task"""

    def __init__(self, system_prompt: tg.Variable, engine):
        self.system_prompt = system_prompt
        self.llm_engine = engine
        # Create the BlackboxLLM with system prompt
        self.model = tg.BlackboxLLM(engine=engine, system_prompt=system_prompt)

    def __call__(self, user_message: tg.Variable) -> tg.Variable:
        """Forward pass through the LLM with current system prompt"""
        return self.model(user_message)

    def parameters(self):
        """Return parameters for the optimizer"""
        return [self.system_prompt]


def create_contextual_compression_loss_fn(query: str, expected_output: str) -> tg.TextLoss:
    """
    Create a TextGrad loss function that evaluates contextual compression quality.
    Returns textual feedback that guides optimization.
    """
    evaluation_instruction = f"""Evaluate the quality of this contextual compression output.

Original Query: {query}
Expected Output: {expected_output}

Evaluation Criteria:
1. Relevance: Does the extracted content directly relate to the query?
2. Completeness: Are all relevant parts extracted without missing important information?
3. Exactness: Is the text copied exactly without modification or paraphrasing?
4. NO_OUTPUT handling: Is NO_OUTPUT used appropriately when no relevant info exists?
5. Format preservation: Are headings and structure preserved correctly?

Provide specific, actionable feedback on how to improve the system prompt for better contextual compression.
Focus on what instructions would help the model extract more relevant content exactly as it appears.
Be constructive and specific about what changes would improve performance."""

    # Create and return TextGrad's TextLoss for evaluation
    return tg.TextLoss(evaluation_instruction)


def evaluate_compression_simple(pred_output: str, target_output: str, has_target: bool) -> float:
    """
    Simple scoring function similar to DSPy optimizer for comparison.
    """
    if has_target:
        if pred_output == "NO_OUTPUT" and target_output != "NO_OUTPUT":
            return 0.0
        elif pred_output == "NO_OUTPUT" and target_output == "NO_OUTPUT":
            return 1.0
        elif target_output != "NO_OUTPUT" and pred_output != "NO_OUTPUT":
            # Simple length-based similarity as proxy
            target_len = len(target_output.strip())
            pred_len = len(pred_output.strip())
            if target_len > 0:
                length_ratio = min(pred_len, target_len) / max(pred_len, target_len)
                return length_ratio * 0.8  # Max score 0.8 for having content
            else:
                return 0.0
        else:
            return 0.0
    else:
        return 0.0


class TextGradGepaOptimizer:
    """TextGrad optimizer that starts from latest GEPA-optimized prompt."""
    
    def __init__(self, observations: List[Dict], gepa_optimized_prompt: str):
        self.observations = observations
        self.gepa_optimized_prompt = gepa_optimized_prompt
        self.setup_textgrad()
        self.setup_weave()
    
    def setup_textgrad(self):
        """Configure TextGrad with engines."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Use GPT-4o-mini as the target model we want to optimize for
        self.target_engine = tg.get_engine('gpt-4o-mini')
        # Use GPT-4o as the critic for gradients
        self.critic_engine = tg.get_engine('gpt-4o')
        tg.set_backward_engine(self.critic_engine)
        
        logger.info("TextGrad configured with GPT-4o-mini target and GPT-4o critic")
    
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
    
    def prepare_examples(self, max_examples: Optional[int] = None) -> List[Dict]:
        """Convert observations to training examples."""
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
            # Create user message template
            user_message = f"""Here is the context document:

<context>
{obs["context"]}
</context>

Now, consider the following query:

<query>
{obs["query"]}
</query>

Now, proceed with the task using the provided context and query."""

            example = {
                "user_message": user_message,
                "context": obs["context"],
                "query": obs["query"],
                "has_target": obs["has_target"],
                "target_output": obs.get("target_output", ""),
                "original_output": obs["original_output"]
            }
            
            examples.append(example)
        
        logger.info(f"Prepared {len(examples)} examples for optimization")
        return examples
    
    def evaluate_model_on_examples(self, model: ContextualCompressionModel, examples: List[Dict],
                                  max_examples: int = None) -> List[float]:
        """Evaluate model performance on a set of examples"""
        if max_examples is None:
            max_examples = len(examples)

        scores = []

        logger.info(f"Evaluating model on {min(max_examples, len(examples))} examples...")

        for i, example in enumerate(tqdm(examples[:max_examples], desc="Evaluating")):
            try:
                # Create TextGrad variable
                user_msg = tg.Variable(
                    example['user_message'],
                    requires_grad=False,
                    role_description="user input for contextual compression"
                )

                # Get model prediction
                prediction = model(user_msg)

                # Simple scoring using the same logic as DSPy optimizer
                pred_output = prediction.value.strip()
                target_output = example.get('target_output', '').strip()
                has_target = example.get('has_target', False)

                score = evaluate_compression_simple(pred_output, target_output, has_target)
                scores.append(score)

            except Exception as e:
                logger.warning(f"Error evaluating example {i+1}: {e}")
                scores.append(0.0)

        return scores
    
    @weave.op()
    def run_optimization(
        self, 
        max_examples: int = 100,
        num_iterations: int = 8,
        batch_size: int = 5,
        train_split: float = 0.8
    ) -> Dict:
        """Run TextGrad optimization starting from GEPA-optimized prompt."""
        
        # Prepare examples
        examples = self.prepare_examples(max_examples)
        
        # Split into train/validation
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        logger.info(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")
        
        # Initialize TextGrad system prompt variable starting from GEPA prompt
        logger.info("Initializing TextGrad system prompt from GEPA-optimized prompt...")
        system_prompt = tg.Variable(
            self.gepa_optimized_prompt,
            requires_grad=True,
            role_description="GEPA-optimized system prompt for contextual compression to be further refined by TextGrad"
        )
        
        # Initialize model
        model = ContextualCompressionModel(system_prompt, self.target_engine)
        
        # Initialize optimizer (TextGrad's TGD - Textual Gradient Descent)
        optimizer = tg.TGD(parameters=[system_prompt])
        
        # Test initial model performance (with GEPA prompt)
        logger.info("Testing initial GEPA-optimized prompt performance...")
        initial_scores = self.evaluate_model_on_examples(model, val_examples[:10], max_examples=10)
        initial_avg = np.mean(initial_scores) if initial_scores else 0.0
        logger.info(f"Initial GEPA-optimized prompt score: {initial_avg:.3f}")
        
        # TextGrad optimization loop
        logger.info("Running TextGrad optimization on GEPA-optimized prompt...")
        logger.info("Using textual gradients to further refine the GEPA prompt...")
        
        best_score = initial_avg
        best_prompt = system_prompt.value
        results = {
            "iteration": [],
            "score": [],
            "prompt": [],
            "best_score": initial_avg
        }
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Sample training examples for this iteration
            train_subset = np.random.choice(
                len(train_examples),
                min(batch_size, len(train_examples)),
                replace=False
            )
            
            iteration_losses = []
            
            # Process batch of examples
            for idx in train_subset:
                example = train_examples[idx]
                
                try:
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Create variables
                    user_msg = tg.Variable(
                        example['user_message'],
                        requires_grad=False,
                        role_description="user input for contextual compression"
                    )
                    
                    # Forward pass
                    prediction = model(user_msg)
                    
                    # Create loss function for this example
                    loss_fn = create_contextual_compression_loss_fn(
                        example['query'],
                        example['target_output']
                    )
                    
                    # Calculate loss
                    loss = loss_fn(prediction)
                    
                    iteration_losses.append(loss)
                    
                    # Backward pass to compute textual gradients
                    loss.backward()
                    
                except Exception as e:
                    logger.warning(f"Error in iteration {iteration + 1}, example {idx}: {e}")
                    continue
            
            if iteration_losses:
                # Apply optimization step (this will update the system prompt)
                optimizer.step()
                
                logger.info(f"Processed {len(iteration_losses)} examples in this iteration")
                
                # Evaluate on validation set
                val_scores = self.evaluate_model_on_examples(model, val_examples[:10], max_examples=10)
                current_score = np.mean(val_scores) if val_scores else 0.0
                
                logger.info(f"Current validation score: {current_score:.3f}")
                
                # Track results
                results["iteration"].append(iteration + 1)
                results["score"].append(current_score)
                results["prompt"].append(system_prompt.value)
                
                # Update best if improved
                if current_score > best_score:
                    best_score = current_score
                    best_prompt = system_prompt.value
                    logger.info(f"✅ New best score: {best_score:.3f}")
                    results["best_score"] = best_score
                else:
                    logger.info(f"Score: {current_score:.3f} (best: {best_score:.3f})")
            
            # Add small delay to avoid rate limits
            time.sleep(2)
        
        # Final evaluation
        final_scores = self.evaluate_model_on_examples(model, val_examples, max_examples=len(val_examples))
        final_avg = np.mean(final_scores) if final_scores else 0.0
        
        optimization_results = {
            "gepa_initial_avg": initial_avg,
            "final_avg": final_avg,
            "best_score": best_score,
            "best_prompt": best_prompt,
            "gepa_starting_prompt": self.gepa_optimized_prompt,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "num_iterations": num_iterations,
            "batch_size": batch_size,
            "optimization_history": results
        }
        
        logger.info(f"Final validation accuracy: {final_avg:.3f}")
        logger.info(f"Best score achieved: {best_score:.3f}")
        logger.info(f"Improvement over GEPA: {best_score - initial_avg:.3f}")
        
        return {
            "model": model,
            "results": optimization_results,
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
    
    # Find and load latest GEPA result
    logger.info("Finding latest GEPA optimization result...")
    latest_gepa_dir = find_latest_gepa_result()
    if not latest_gepa_dir:
        logger.error("No GEPA results found. Run GEPA optimization first with 'make optimize'")
        return 1
    
    gepa_optimized_prompt = load_gepa_optimized_prompt(latest_gepa_dir)
    if not gepa_optimized_prompt:
        logger.error("Could not load GEPA-optimized prompt")
        return 1
    
    logger.info(f"Starting TextGrad optimization from GEPA prompt: {latest_gepa_dir.name}")
    
    # Load data
    logger.info("Loading observation data...")
    loader = DataLoader(str(observations_dir), str(gpt4o_dir))
    observations = loader.load_observations()
    
    if not observations:
        raise ValueError("No valid observations found")
    
    # Run optimization
    optimizer = TextGradGepaOptimizer(observations, gepa_optimized_prompt)
    
    # Start with a smaller subset for initial testing
    optimization_results = optimizer.run_optimization(
        max_examples=50,  # Start small
        num_iterations=8,
        batch_size=5,
        train_split=0.8
    )
    
    # Save results with timestamp and specific naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = project_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific subdirectory
    experiment_name = f"textgrad_gepa_context_compression_{timestamp}"
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimized prompt
    prompt_file = experiment_dir / "optimized_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(optimization_results["results"]["best_prompt"])
    
    # Save detailed results with experiment parameters
    results_with_metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "script": "textgrad_latest_gepa_optimizer.py",
        "optimization_type": "TextGrad TGD on GEPA",
        "model_target": "gpt-4o-mini",
        "critic_model": "gpt-4o",
        "gepa_source": str(latest_gepa_dir),
        "gepa_starting_prompt": optimization_results["results"]["gepa_starting_prompt"],
        "optimized_prompt": optimization_results["results"]["best_prompt"],
        "parameters": {
            "max_examples": 50,
            "num_iterations": 8,
            "batch_size": 5,
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
            "gepa_source": str(latest_gepa_dir),
            "gepa_initial_accuracy": optimization_results["results"]["gepa_initial_avg"],
            "final_accuracy": optimization_results["results"]["final_avg"],
            "best_accuracy": optimization_results["results"]["best_score"],
            "improvement_over_gepa": optimization_results["results"]["best_score"] - optimization_results["results"]["gepa_initial_avg"],
            "script": "textgrad_latest_gepa_optimizer.py"
        }
        json.dump(summary, f, indent=2)
    
    logger.info(f"Optimization complete! Results saved to {experiment_dir}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"GEPA starting accuracy: {optimization_results['results']['gepa_initial_avg']:.3f}")
    logger.info(f"Final accuracy: {optimization_results['results']['final_avg']:.3f}")
    logger.info(f"Best accuracy: {optimization_results['results']['best_score']:.3f}")
    logger.info(f"Improvement over GEPA: {optimization_results['results']['best_score'] - optimization_results['results']['gepa_initial_avg']:.3f}")


if __name__ == "__main__":
    main()