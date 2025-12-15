# local_llm.py

import logging
from ctransformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class LocalLLM:
    """
    Runs a language model locally using ctransformers.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "llama",
        context_length: int = 2048,
        threads: int = 4
    ):
        logger.info(f"Loading model: {model_path}")
        
        # Auto-detect model type from filename
        model_path_lower = model_path.lower()
        if "mistral" in model_path_lower:
            model_type = "mistral"
        elif "llama" in model_path_lower or "tinyllama" in model_path_lower:
            model_type = "llama"
        
        logger.info(f"Model type: {model_type}")
        
        # Load the model
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            context_length=context_length,
            threads=threads
        )
        
        logger.info("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from a prompt.
        """
        output = self.llm(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "Human:", "User:", "\n\n\n"]
        )
        
        return output.strip()