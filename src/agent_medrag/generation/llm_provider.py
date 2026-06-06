'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-05 14:24:05
LastEditTime: 2026-06-05 20:06:52
FilePath: /Agent_MedRag/src/agent_medrag/generation/llm_provider.py
Description: 

'''
from __future__ import annotations

import os
from typing import Any,Protocol

class LLMProvider(Protocol):
    def generate(self,prompt:str)->str:
        '''
        Generate text from particular prompt
        '''
        ...
        
class DeepSeekProvider:
    def __init__(
        self,
        model:str='deepseek-v4-pro',
        temperature:float=0.1,
        max_output_tokens:int=800,
        thinking_enabled:bool=False,
        reasoning_effort:str='high',
        client:Any | None=None,
    )->None:
        if not model.strip():
            raise ValueError("please appoint a model.")
        
        if not 0<=temperature<=2:
            raise ValueError("temperature must be between 0 and 2.")

        if max_output_tokens<=0:
            raise ValueError("max_output_tokens must be positive.")
        
        if reasoning_effort not in {'high','max'}:
            raise ValueError(
                "reasoning_effort must be 'high' or 'max'."
            )
        
        if client is None:
            if not os.getenv('DEEPSEEK_API_KEY'):
                raise ValueError(
                    "DEEPSEEK_API_KEY environment variable is not set."
                )

            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required. "
                    "Install it with: pip install openai"
                ) from exc

            client=OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com",
            )
            
        self.client=client
        self.model=model
        self.temperature=temperature
        self.max_output_tokens=max_output_tokens
        self.thinking_enabled=thinking_enabled
        self.reasoning_effort=reasoning_effort
        
    def generate(self,prompt:str)->str:
        cleaned_prompt=prompt.strip()
        
        if not cleaned_prompt:
            raise ValueError("prompt must not be empty.")
        
        request_options:dict[str,Any]={
            "model":self.model,
            "messages":[
                {
                    'role':'user',
                    'content':cleaned_prompt,
                }
            ],
            'max_tokens':self.max_output_tokens,
            'stream':False,
            'extra_body':{
                'thinking':{
                    'type':('enabled' if self.thinking_enabled else 'disabled')
                }
            },
        }
        
        if self.thinking_enabled:
            request_options['reasoning_effort']=self.reasoning_effort
        else:
            request_options['temperature']=self.temperature
        
        response=self.client.chat.completions.create(
            **request_options
        )
        
        output_text=response.choices[0].message.content
        
        if not output_text or not output_text.strip():
            raise RuntimeError("DeepSeek returned an empty response.")
        
        return output_text.strip()