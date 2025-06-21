"""
LLM processing module for generating responses based on retrieved content
"""

from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from transformers.utils import is_flash_attn_2_available
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rag_pipeline.core.config import AppConfig, PipelineStage
from rag_pipeline.utils.device import get_device, get_model_memory_info
from rag_pipeline.utils.text import clean_generated_text
from rag_pipeline.utils.logging import get_logger

console = Console()
logger = get_logger()


class LLMProcessor:
    """Manages LLM for generating responses based on retrieved content"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_stage = PipelineStage.LLM_SETUP
        self.tokenizer = None
        self.llm_model = None
        self.device = get_device(config.llm_device)
        
    def initialize_model(self) -> None:
        """Initialize LLM and tokenizer"""
        try:
            console.print(f"[yellow]Loading LLM: {self.config.llm_model_id}...[/yellow]")
            
            # Create quantization config if needed
            quantization_config = None
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                console.print("[yellow]Using 4-bit quantization[/yellow]")
            
            # Setup attention implementation
            if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
                attn_implementation = "flash_attention_2"
                console.print("[green]Using Flash Attention 2[/green]")
            else:
                attn_implementation = "sdpa"
                console.print("[yellow]Using scaled dot product attention[/yellow]")
            
            # Instantiate tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.config.llm_model_id
            )
            
            # Instantiate the model
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.llm_model_id,
                torch_dtype=torch.float16,  # datatype to use
                quantization_config=quantization_config if self.config.use_quantization else None,
                low_cpu_mem_usage=False,  # use full memory
                attn_implementation=attn_implementation  # which attention version to use
            )
            
            if not self.config.use_quantization:
                self.llm_model.to(self.device)
            
            # Model stats
            memory_info = get_model_memory_info(self.llm_model)
            
            console.print(f"[green]LLM loaded successfully:[/green]")
            console.print(f"[blue]Parameters: {memory_info['num_parameters']:,}[/blue]")
            console.print(f"[blue]Memory usage: {memory_info['memory_gb']:.2f} GB[/blue]")
            
        except Exception as e:
            console.print(f"[red]Error initializing LLM: {e}[/red]")
            raise
    
    def prompt_formatter(self, query: str, context_items: List[Dict]) -> str:
        """Format prompt with query and context items"""
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        # Create a base prompt with examples
        base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
             "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation=dialogue_template,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    def generate_answer(self, 
                       query: str, 
                       context_items: List[Dict],
                       temperature: float = None,
                       max_new_tokens: int = None) -> str:
        """
        Generate an answer based on the query and context items
        
        Args:
            query: The user query
            context_items: List of relevant context items
            temperature: Generation temperature (randomness)
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated answer text
        """
        if temperature is None:
            temperature = self.config.temperature
            
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
            
        try:
            # Format the prompt
            prompt = self.prompt_formatter(query=query, context_items=context_items)
            
            # Tokenize the prompt
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating answer..."),
                BarColumn(),
                console=console
            ) as progress:
                progress.add_task("Generating", total=1)
                
                # Generate output
                outputs = self.llm_model.generate(
                    **input_ids,
                    temperature=temperature,
                    do_sample=True,
                    max_new_tokens=max_new_tokens
                )
            
            # Decode output
            output_text = self.tokenizer.decode(outputs[0])
            
            # Clean output
            formatted_answer = clean_generated_text(output_text, prompt)
            
            return formatted_answer
            
        except Exception as e:
            console.print(f"[red]Error generating answer: {e}[/red]")
            raise
