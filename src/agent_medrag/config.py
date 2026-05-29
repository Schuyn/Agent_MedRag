'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-16 18:06:40
LastEditTime: 2026-05-29 15:02:31
FilePath: /Agent_MedRag/src/agent_medrag/config.py
Description:
Current functionality:
- Load YAML configuration files
- Provide safe configuration access helpers
- Perform minimal Stage 2 validation
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

DAFAULT_CONFIG_PATH="configs/default.yaml"

def load_config(config_path: str | Path=DAFAULT_CONFIG_PATH)->dict[str,Any]:
    '''
    Load a YAML configuration file and return it as a dictionary.
    '''
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config files. Install it with: pip install PyYAML"
        ) from exc
        
    path=Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r",encoding='utf-8') as file:
        config=yaml.safe_load(file) or {}
        
    if not isinstance(config,dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
        
    validate_stage2_config(config)
    return config

def get_config_value(
    config: dict[str,Any],
    section: str,
    key: str,
    default: Any=None,
)->Any:
    '''
    Safely retrieve a nested configuration value.
    '''
    section_data=config.get(section,{})
    
    if not isinstance(section_data,dict):
        return default
    
    return section_data.get(key,default)

def resolve_device(device: str | None)->str | None:
    '''
    Normalize the device parameter.
    '''
    if device is None:
        return None
    
    if device.lower()=='auto':
        return None
    
    return None

def validate_stage2_config(config:dict[str,Any])->None:
    '''
    Perform the minimal configuration validation required for Stage 2.
    '''
    chunk_size=get_config_value(config,"chunking","chunk_size",1000)
    chunk_overlap=get_config_value(config,"chunking","chunk_overlap",150)
    batch_size=get_config_value(config,"embedding","batch_size",64)
    
    if chunk_size<=0:
        raise ValueError("chunking.chunk_size must be positive.")
    
    if chunk_overlap<0 or chunk_overlap>=chunk_size:
        raise ValueError("chunking.chunk_overlap must be non-negative and smaller than chunk_size.")
    
    if batch_size<=0:
        raise ValueError("embedding.batch_size must be positive.")
    
    embedding_provider=get_config_value(config,"embedding","provider","huggingface")
    if embedding_provider!="huggingface":
        raise ValueError("Stage 2 only supports embedding.provider='huggingface'.")

    vector_provider=get_config_value(config,"vectore_store","provider","chroma")
    if vector_provider!="chroma":
        raise ValueError("Stage 2 only supports vector_store.provider='chroma'.")