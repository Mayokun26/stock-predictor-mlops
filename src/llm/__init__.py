"""
LLM Integration Package for Stock Prediction MLOps System

Provides AI-powered explanations, market research, and risk assessment
for stock predictions using OpenAI and other language models.
"""

from .llm_service import LLMService, LLMRequest, LLMResponse, PromptTemplate, get_llm_service

__all__ = [
    'LLMService',
    'LLMRequest', 
    'LLMResponse',
    'PromptTemplate',
    'get_llm_service'
]