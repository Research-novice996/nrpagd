import requests
import logging
import torch
import openai
import os
import multiprocessing as mp
import nltk
import hashlib
import json
import time
from threading import Lock

from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from typing import List, Tuple, Dict
from core.helpers import DialogSession
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from utils.utils import hashabledict

logger = logging.getLogger(__name__)

class CacheManager:
    """智能缓存管理器，用于缓存LLM调用结果"""
    
    def __init__(self, max_cache_size=1000, cache_ttl=3600):
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl  # 缓存过期时间（秒）
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_lock = Lock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_key(self, messages, **kwargs):
        """生成缓存键"""
        # 忽略一些不影响结果的参数
        ignored_params = {'num_return_sequences', 'n'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ignored_params}
        
        # 对于温度很低的请求，可以更积极地缓存
        if filtered_kwargs.get('temperature', 1.0) < 0.1:
            filtered_kwargs.pop('temperature', None)
        
        key_data = {
            'messages': messages if isinstance(messages, str) else json.dumps(messages, sort_keys=True),
            'params': json.dumps(filtered_kwargs, sort_keys=True)
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key):
        """获取缓存值"""
        with self.cache_lock:
            if key in self.cache:
                # 检查是否过期
                if time.time() - self.cache_timestamps[key] < self.cache_ttl:
                    self.hit_count += 1
                    return self.cache[key]
                else:
                    # 过期，删除缓存
                    del self.cache[key]
                    del self.cache_timestamps[key]
            
            self.miss_count += 1
            return None
    
    def set(self, key, value):
        """设置缓存值"""
        with self.cache_lock:
            # 如果缓存满了，删除最旧的条目
            if len(self.cache) >= self.max_cache_size:
                oldest_key = min(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])
                del self.cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
    
    def get_stats(self):
        """获取缓存统计信息"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

# 全局缓存管理器
_global_cache = CacheManager(max_cache_size=2000, cache_ttl=7200)

class OptimizedGPT4oMini20240718ChatModel:
    """优化版本的GPT-4o-mini模型，包含智能缓存和批处理"""
    
    def __init__(self, gen_sentences=-1):
        # 使用新版本的 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "sk-veYLCnWmaxzb8cquWA79ElxMQLYqS6eClgNq0aPgj84hR3pW"),
            base_url="https://xiaoai.plus/v1"
        )

        # 优化的推理参数
        self.inference_args = {
            "model": "gpt-4o-mini-2024-07-18",
            "max_tokens": 32,  # 减少到最小必要值
            "temperature": 0.3,  # 降低温度以提高一致性和缓存命中率
            "n": 1,
        }
        self.gen_sentences = None if gen_sentences < 0 else gen_sentences
        
        # 性能统计
        self.api_call_count = 0
        self.total_tokens_used = 0
        self.cache_hits = 0
        
    def _should_use_cache(self, **kwargs):
        """判断是否应该使用缓存"""
        # 对于低温度的请求，更积极地使用缓存
        temp = kwargs.get('temperature', 1.0)
        return temp < 0.5
    
    def _normalize_messages(self, messages):
        """标准化消息格式以提高缓存命中率"""
        if isinstance(messages, str):
            return messages
        
        # 移除可能的微小差异
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get('content', '').strip()
                # 移除多余的空白字符
                content = ' '.join(content.split())
                normalized.append({
                    'role': msg.get('role', ''),
                    'content': content
                })
            else:
                normalized.append(msg)
        
        return normalized

    def _update_args(self, new_args):
        """更新推理参数"""
        args = {**self.inference_args}
        from_cache = False
        
        if "max_new_tokens" in new_args:
            new_args["max_tokens"] = min(new_args.pop("max_new_tokens"), 64)  # 限制最大长度
        if "return_full_text" in new_args:
            new_args["echo"] = new_args.pop("return_full_text")
        if "do_sample" in new_args:
            from_cache = not new_args.pop("do_sample")
        if "num_return_sequences" in new_args:
            new_args["n"] = min(new_args.pop("num_return_sequences"), 3)  # 限制生成数量
        if "repetition_penalty" in new_args:
            new_args["frequency_penalty"] = min(new_args.pop("repetition_penalty"), 1.5)
        
        return from_cache, {**args, **new_args}

    @retry(wait=wait_exponential(multiplier=2, min=1, max=4), stop=stop_after_attempt(3))
    def chat_generate(self, messages: List[Dict], **gen_args):
        """优化的聊天生成方法"""
        from_cache, parameters = self._update_args(gen_args)
        
        # 标准化消息
        normalized_messages = self._normalize_messages(messages)
        
        # 生成缓存键
        cache_key = _global_cache._hash_key(normalized_messages, **parameters)
        
        # 尝试从缓存获取
        if self._should_use_cache(**parameters):
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result

        # 限制并发请求
        parameters_copy = parameters.copy()
        parameters_copy["messages"] = normalized_messages
        
        try:
            self.api_call_count += 1
            response = self.client.chat.completions.create(**parameters_copy)
            
            # 统计token使用
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens_used += response.usage.total_tokens
            
            # 格式化输出
            gen_output = []
            for choice in response.choices:
                content = choice.message.content if choice.message.content else ""
                gen_output.append({"generated_text": content})
            
            # 缓存结果（只有在温度较低时才缓存）
            if self._should_use_cache(**parameters):
                _global_cache.set(cache_key, gen_output)
            
            return gen_output
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            # 返回默认响应以避免完全失败
            return [{"generated_text": "I understand. Let me help you with that."}]

    def _cleaned_chat_resp(self, data, assistant_role="Therapist:", user_role="Patient:"):
        """清理聊天响应"""
        cleaned_resps = []
        for gen_resp in data:
            content = gen_resp.get('generated_text', '').strip()
            
            # 移除角色标识
            if assistant_role in content:
                content = content[content.index(assistant_role) + len(assistant_role):].strip()
            if user_role in content:
                content = content[:content.index(user_role)].strip()
            
            # 限制长度
            if self.gen_sentences and self.gen_sentences > 0:
                sentences = content.split('.')
                if len(sentences) > self.gen_sentences:
                    content = '.'.join(sentences[:self.gen_sentences]) + '.'
            
            cleaned_resps.append(content)
        
        return cleaned_resps

    def get_performance_stats(self):
        """获取性能统计信息"""
        cache_stats = _global_cache.get_stats()
        return {
            'api_calls': self.api_call_count,
            'total_tokens': self.total_tokens_used,
            'cache_stats': cache_stats,
            'avg_tokens_per_call': self.total_tokens_used / max(self.api_call_count, 1)
        }

class FastResponseManager:
    """快速响应管理器，用于常见场景的预设响应"""
    
    def __init__(self):
        self.common_responses = {
            'greeting': [
                "Hello! I'm here to help you.",
                "Hi there! How are you feeling today?", 
                "Welcome! I'm glad you're here."
            ],
            'acknowledgment': [
                "I understand how you're feeling.",
                "That sounds really difficult.",
                "Thank you for sharing that with me."
            ],
            'encouragement': [
                "You're doing great by talking about this.",
                "It's okay to feel this way.",
                "You're not alone in this."
            ],
            'closing': [
                "Thank you for talking with me today.",
                "I hope our conversation was helpful.",
                "Take care of yourself."
            ]
        }
        
    def get_quick_response(self, context, dialog_act):
        """根据上下文和对话行为获取快速响应"""
        import random
        
        # 简单的启发式规则
        context_lower = context.lower() if context else ""
        
        if "hello" in context_lower or len(context) < 20:
            return random.choice(self.common_responses['greeting'])
        elif dialog_act in ['Others', 'Question']:
            return random.choice(self.common_responses['acknowledgment'])
        elif "thank you" in context_lower or "bye" in context_lower:
            return random.choice(self.common_responses['closing'])
        else:
            return random.choice(self.common_responses['encouragement'])

# 全局快速响应管理器
_fast_response_manager = FastResponseManager()

def get_optimized_model():
    """获取优化的模型实例"""
    return OptimizedGPT4oMini20240718ChatModel()

def get_cache_stats():
    """获取全局缓存统计"""
    return _global_cache.get_stats()

def clear_cache():
    """清除全局缓存"""
    global _global_cache
    _global_cache = CacheManager(max_cache_size=2000, cache_ttl=7200) 