#!/usr/bin/env python3
"""
Test the latest GEPA-optimized prompt against all observations.
This script loads the most recent GEPA optimization result, extracts the optimized prompt,
and tests it against all observations using OpenAI GPT-4o-mini.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def load_optimized_prompt(gepa_dir: Path) -> Optional[str]:
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
        
        logger.info(f"Loaded optimized prompt ({len(instructions)} chars)")
        logger.info(f"Prompt preview: {instructions[:200]}...")
        return instructions
        
    except Exception as e:
        logger.error(f"Error loading optimized model: {e}")
        return None

def extract_context_and_query(input_messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """Extract context and query from observation input messages."""
    context = None
    query = None
    
    for message in input_messages:
        content = message.get("content", "")
        
        # Extract context from <context>...</context> tags
        if not context:
            context_match = re.search(r'<context>(.*?)</context>', content, re.DOTALL)
            if context_match:
                context = context_match.group(1).strip()
        
        # Extract query from <query>...</query> tags
        if not query:
            query_match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
            if query_match:
                query = query_match.group(1).strip()
    
    return context, query

def load_observations() -> List[Dict]:
    """Load all observation files from data/observations/."""
    observations_dir = Path(__file__).parent.parent / "data" / "observations"
    
    if not observations_dir.exists():
        logger.error(f"Observations directory not found: {observations_dir}")
        return []
    
    observations = []
    observation_files = [f for f in observations_dir.glob("*.json") if not f.name.startswith("_")]
    
    logger.info(f"Loading {len(observation_files)} observation files...")
    
    for obs_file in tqdm(observation_files, desc="Loading observations"):
        try:
            with open(obs_file, 'r') as f:
                obs_data = json.load(f)
            
            # Extract context and query from input messages
            input_messages = obs_data.get("input", [])
            context, query = extract_context_and_query(input_messages)
            
            if context and query:
                observations.append({
                    "id": obs_data.get("id", obs_file.stem),
                    "context": context,
                    "query": query,
                    "original_output": obs_data.get("output", {}).get("content", ""),
                    "file_path": obs_file
                })
            else:
                logger.warning(f"Could not extract context/query from {obs_file.name}")
                
        except Exception as e:
            logger.error(f"Error loading {obs_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(observations)} observations")
    return observations

def create_test_prompt(optimized_instructions: str, context: str, query: str) -> str:
    """Create the test prompt with optimized instructions and context/query."""
    user_message = f"""Here is the context document:

<context>
{context}
</context>

Now, consider the following query:

<query>
{query}
</query>

Now, proceed with the task using the provided context and query."""

    return user_message

def test_observation_with_openai(client: openai.OpenAI, optimized_instructions: str, observation: Dict) -> Dict:
    """Test a single observation using OpenAI GPT-4o-mini with optimized prompt."""
    user_message = create_test_prompt(optimized_instructions, observation["context"], observation["query"])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": optimized_instructions},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        
        output = response.choices[0].message.content.strip()
        success = bool(output and output != "NO_OUTPUT")
        
        return {
            "id": observation["id"],
            "success": success,
            "output": output,
            "input_context": observation["context"][:200] + "..." if len(observation["context"]) > 200 else observation["context"],
            "input_query": observation["query"],
            "original_output": observation["original_output"],
            "usage": response.usage.model_dump() if response.usage else None
        }
        
    except Exception as e:
        logger.error(f"Error testing observation {observation['id']}: {e}")
        return {
            "id": observation["id"],
            "success": False,
            "output": f"ERROR: {str(e)}",
            "input_context": observation["context"][:200] + "..." if len(observation["context"]) > 200 else observation["context"],
            "input_query": observation["query"],
            "original_output": observation["original_output"],
            "error": str(e)
        }

def main():
    logger.info("Testing latest GEPA-optimized prompt")
    logger.info("=" * 50)
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return 1
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Find latest GEPA result
    logger.info("1. Finding latest GEPA optimization result...")
    latest_gepa_dir = find_latest_gepa_result()
    if not latest_gepa_dir:
        return 1
    
    # Load optimized prompt
    logger.info("2. Loading optimized prompt...")
    optimized_instructions = load_optimized_prompt(latest_gepa_dir)
    if not optimized_instructions:
        return 1
    
    # Load observations
    logger.info("3. Loading observations...")
    observations = load_observations()
    if not observations:
        logger.error("No observations loaded")
        return 1
    
    # Create test results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path(__file__).parent.parent / "data" / "tests" / f"gpt-4o-mini-test-{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results subfolder for individual test results
    results_dir = test_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"4. Created test directory: {test_dir}")
    logger.info(f"   Created results subdirectory: {results_dir}")
    
    # Save test configuration
    config = {
        "timestamp": timestamp,
        "gepa_source": str(latest_gepa_dir),
        "optimized_instructions": optimized_instructions,
        "total_observations": len(observations),
        "model": "gpt-4o-mini-2024-07-18"
    }
    
    with open(test_dir / "test_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test all observations
    logger.info(f"5. Testing {len(observations)} observations with optimized prompt...")
    results = []
    successful_tests = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for i, observation in enumerate(tqdm(observations, desc="Testing observations")):
        result = test_observation_with_openai(client, optimized_instructions, observation)
        results.append(result)
        
        if result["success"]:
            successful_tests += 1
        
        # Accumulate token usage
        if result.get("usage"):
            total_usage["prompt_tokens"] += result["usage"].get("prompt_tokens", 0)
            total_usage["completion_tokens"] += result["usage"].get("completion_tokens", 0)
            total_usage["total_tokens"] += result["usage"].get("total_tokens", 0)
        
        # Save individual result in results subfolder
        result_file = results_dir / f"{observation['id']}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Calculate success rate
    success_rate = (successful_tests / len(observations)) * 100 if observations else 0
    
    # Save summary results
    summary = {
        "timestamp": timestamp,
        "gepa_source": str(latest_gepa_dir),
        "total_observations": len(observations),
        "successful_tests": successful_tests,
        "failed_tests": len(observations) - successful_tests,
        "success_rate_percentage": round(success_rate, 2),
        "total_usage": total_usage,
        "model": "gpt-4o-mini-2024-07-18"
    }
    
    summary_file = test_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Log results
    logger.info("=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total observations tested: {len(observations)}")
    logger.info(f"Successful tests: {successful_tests}")
    logger.info(f"Failed tests: {len(observations) - successful_tests}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Total tokens used: {total_usage['total_tokens']:,}")
    logger.info(f"Results saved to: {test_dir}")
    logger.info("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit(main())