import ollama
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    json_config: Optional[Dict[str, Any]] = None

class LocalLLM:
    """
    Wrapper for local Llama 3.2 via ChatOllama.
    Ensures consistent prompt formatting for quant tasks.
    """
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0} # Determinism is key for Quant
        )
        return response['message']['content']

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Forces the LLM to output valid JSON.
        In a production environment, this would use constrained sampling (JSON mode).
        """
        # Append strict JSON requirement to the prompt
        enhanced_prompt = f"{user_prompt}\n\nIMPORTANT: Return ONLY a valid JSON object. No markdown, no explanations."
        response = self.generate(system_prompt, enhanced_prompt)

        # Simple extraction of JSON block if LLM includes markdown
        import json
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except Exception as e:
            raise ValueError(f"LLM failed to provide valid JSON: {str(e)}\nResponse: {response}")
