import csv
import os
import functools
from datetime import datetime
from typing import Callable, Any, Dict, Optional
from pathlib import Path

from src.observability.logging_config import get_logger

logger = get_logger(__name__)

# Pricing configuration (USD per 1M tokens)
# Update these values as needed
PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "cohere": {
        "command-r-plus": {"input": 3.00, "output": 15.00},
        "command-r": {"input": 0.50, "output": 1.50},
        "command": {"input": 1.50, "output": 2.00},
    }
}

# CSV file path
COST_CSV_PATH = Path("Logs/costs.csv")


class CostTracker:
    """Track estimated token usage and cost for embedding requests in one run."""

    EMBEDDING_COSTS_PER_1K_TOKENS = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.00010,
    }

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if self.model not in self.EMBEDDING_COSTS_PER_1K_TOKENS:
            raise ValueError(f"Unsupported embedding model for cost tracking: {self.model}")
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def track_embedding_cost(self, text_length: int) -> None:
        """Add an estimated embedding charge (approximately four characters per token)."""
        if text_length < 0:
            raise ValueError("text_length cannot be negative")

        estimated_tokens = (text_length + 3) // 4
        self.total_tokens += estimated_tokens
        self.total_cost_usd += (
            estimated_tokens / 1_000 * self.EMBEDDING_COSTS_PER_1K_TOKENS[self.model]
        )

    def get_summary(self) -> Dict[str, float | int]:
        """Return the estimated embedding usage accumulated for this run."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


def _ensure_csv_header() -> None:
    """Ensure the CSV file exists with headers."""
    try:
        COST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not COST_CSV_PATH.exists() or COST_CSV_PATH.stat().st_size == 0:
            with open(COST_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "run_id", "strategy", "input_tokens", "output_tokens", "cost_usd"])
            logger.info("Created costs.csv with headers", extra={"path": str(COST_CSV_PATH)})
    except OSError as e:
        logger.error("Failed to create costs.csv", extra={"error": str(e), "path": str(COST_CSV_PATH)})
        raise


def _calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost based on provider, model, and token usage.
    
    Args:
        provider: API provider (e.g., "openai", "cohere")
        model: Model name (e.g., "gpt-4o", "command-r")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Cost in USD
    
    Raises:
        ValueError: If provider or model not found in pricing config
    """
    try:
        pricing = PRICING[provider.lower()][model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    except KeyError as e:
        logger.warning("Pricing not found for provider/model", extra={
            "provider": provider,
            "model": model,
            "error": str(e)
        })
        raise ValueError(f"Pricing not found for {provider}/{model}") from e


def _append_cost_row(
    run_id: str,
    strategy: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float
) -> None:
    """
    Append a cost row to the CSV file.
    
    Args:
        run_id: Run identifier
        strategy: Strategy name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD
    
    Raises:
        OSError: If file write fails
    """
    try:
        _ensure_csv_header()
        with open(COST_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                run_id,
                strategy,
                input_tokens,
                output_tokens,
                f"{cost_usd:.6f}"
            ])
        logger.info("Cost row appended", extra={
            "run_id": run_id,
            "strategy": strategy,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd
        })
    except OSError as e:
        logger.error("Failed to append cost row", extra={
            "run_id": run_id,
            "error": str(e)
        })
        raise


def track_api_cost(
    provider: str,
    model: str,
    run_id_param: str = "run_id",
    strategy_param: str = "strategy",
    input_tokens_extractor: Optional[Callable[[Any], int]] = None,
    output_tokens_extractor: Optional[Callable[[Any], int]] = None
):
    """
    Decorator to track API costs for functions that call external APIs.
    
    Args:
        provider: API provider (e.g., "openai", "cohere")
        model: Model name (e.g., "gpt-4o", "command-r")
        run_id_param: Name of the parameter containing run_id
        strategy_param: Name of the parameter containing strategy
        input_tokens_extractor: Function to extract input tokens from response (default: response.usage.prompt_tokens)
        output_tokens_extractor: Function to extract output tokens from response (default: response.usage.completion_tokens)
    
    Usage:
        @track_api_cost(provider="openai", model="gpt-4o")
        def call_openai(run_id: str, strategy: str, prompt: str):
            return client.chat.completions.create(...)
    
    For custom token extraction:
        @track_api_cost(
            provider="cohere",
            model="command-r",
            input_tokens_extractor=lambda r: r.meta.billed_units.input_tokens,
            output_tokens_extractor=lambda r: r.meta.billed_units.output_tokens
        )
        def call_cohere(run_id: str, strategy: str, prompt: str):
            return co.generate(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract run_id and strategy from function arguments
            try:
                # Get function signature to map parameter names
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                run_id = bound_args.arguments.get(run_id_param)
                strategy = bound_args.arguments.get(strategy_param)
                
                if not run_id:
                    logger.warning(f"run_id not found in parameter '{run_id_param}'", extra={"function": func.__name__})
                    return func(*args, **kwargs)
                if not strategy:
                    logger.warning(f"strategy not found in parameter '{strategy_param}'", extra={"function": func.__name__})
                    return func(*args, **kwargs)
                
                # Call the original function
                response = func(*args, **kwargs)
                
                # Extract token usage
                if input_tokens_extractor:
                    input_tokens = input_tokens_extractor(response)
                else:
                    # Default extraction for OpenAI-style responses
                    input_tokens = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) if response else 0
                
                if output_tokens_extractor:
                    output_tokens = output_tokens_extractor(response)
                else:
                    # Default extraction for OpenAI-style responses
                    output_tokens = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) if response else 0
                
                # Calculate cost
                cost_usd = _calculate_cost(provider, model, input_tokens, output_tokens)
                
                # Append to CSV
                _append_cost_row(run_id, strategy, input_tokens, output_tokens, cost_usd)
                
                return response
                
            except Exception as e:
                logger.error("Error in cost tracking decorator", extra={
                    "function": func.__name__,
                    "error": str(e)
                })
                # Re-raise to not break the original function
                raise
        
        return wrapper
    return decorator
