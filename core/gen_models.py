import requests
import logging
import torch
import openai
import os
import multiprocessing as mp
import nltk

from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from typing import List, Tuple, Dict
from core.helpers import DialogSession
from functools import lru_cache
from tenacity import retry, stop_after_attempt,	wait_exponential, wait_fixed  # for exponential backoff
from utils.utils import hashabledict


logger = logging.getLogger(__name__)





class GenerationModel(ABC):
	# used to generate text in general. e.g. could be using API, or local model
	@abstractmethod
	def generate(self, input_text, **gen_args):
		"""
		Generate text from the model.
		"""
		raise NotImplementedError

	def chat_generate(self, messages, **gen_args):
		"""
		Generate text from the model. Used for chatbot.
		"""
		raise NotImplementedError
	
	def chat_generate_batched(self, messages_list, **gen_args):
		"""
		Generate text from the model when you have multiple message histories
		"""
		raise NotImplementedError

	def _cleaned_resp(self, data, prompt) -> "List[str]":
		# default helper function to clean extract the generated text from the returned json
		cleaned_resps = []
		for gen_resp in data:
			cleaned_resp = gen_resp['generated_text'].strip()
			if "\n" in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
			cleaned_resps.append(cleaned_resp)
		return cleaned_resps
	
	def _cleaned_chat_resp(self, data, assistant_role="Persuader:", user_role="Persuadee:") -> "List[str]":
		# remove the user_role and keep the assistant_role
		# default helper function to clean extract the generated text from the returned json
		cleaned_resps = []
		for gen_resp in data:
			cleaned_resp = gen_resp['generated_text'].strip()
			if "\n" in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
			if assistant_role in cleaned_resp:
				cleaned_resp = cleaned_resp[cleaned_resp.index(assistant_role) + len(assistant_role):].strip()
			if user_role in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index(user_role)].strip()
			cleaned_resps.append(cleaned_resp)
		return cleaned_resps


class DialogModel(ABC):
	# used to play DialogGame
	def __init__(self):
		self.dialog_acts = []
		return
	
	@abstractmethod
	def get_utterance(self, state:DialogSession, action) -> str:
		raise NotImplementedError
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int) -> List[str]:
		raise NotImplementedError

	@abstractmethod
	def get_utterance_w_da(self, state:DialogSession, action) -> Tuple[str, str]:
		# this is used for user agent. should not be used for system agent
		raise NotImplementedError
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		# this is used for user agent. should not be used for system agent
		raise NotImplementedError
		


class APIModel(GenerationModel):
	API_TOKEN = os.environ.get("HF_API_KEY")

	def __init__(self):
		# self.API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
		self.API_URL = "https://api-inference.huggingface.co/models/gpt2-large"
		self.headers: dict[str, str] = {"Authorization": f"Bearer {APIModel.API_TOKEN}"}
		self.inference_args = {
			"max_new_tokens": 100,
			"temperature": 0.7,
			"repetition_penalty": 1.2,
			"return_full_text": False
		}
		return

	def generate(self, input_text, **_args):
		data = {
			"inputs": input_text,
			"parameters": _args or self.inference_args
		}
		response = requests.post(self.API_URL, headers=self.headers, json=data)
		return response.json()


class OpenAIModel(GenerationModel):
	def __init__(self, model_name="gpt-3.5-turbo-0613"):
		# 使用新版本的 OpenAI 客户端
		self.client = openai.OpenAI(
			api_key=os.getenv("OPENAI_API_KEY", "sk-cucQ104SzP9t8h2zge5wRzsPUKZe8BD4GDz9AOJsGhuJHhdx"),
			base_url="https://xiaoai.plus/v1/chat/completions"
		)

		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"echo": False,
			"n": 1,
			"stop": "\n"
		}
		return

	def _update_args(self, new_args):
		args = {**self.inference_args}
		from_cache = False
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "return_full_text" in new_args:
			new_args["echo"] = new_args.pop("return_full_text")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")  # rely on caching
		if "num_return_sequences" in new_args:
			new_args["n"] = new_args.pop("num_return_sequences")
		if "repetition_penalty" in new_args:
			new_args["frequency_penalty"] = new_args.pop("repetition_penalty")
		return from_cache, {**args, **new_args}

	@lru_cache(maxsize=None)
	def _cached_generate(self, **parameters):
		response = self.client.completions.create(**parameters)
		return response

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def generate(self, input_text, **_args):
		from_cache, parameters = self._update_args(_args)
		parameters["prompt"] = input_text
		if from_cache:
			response = self._cached_generate(**parameters)
		else:
			response = self.client.completions.create(**parameters)

		# 格式化输出
		gen_output = []
		for resp in response.choices:
			text = resp.text
			gen_output.append({"generated_text": text})
		return gen_output


class OpenAIChatModel(OpenAIModel):
	"""
    Chat Model 基于 OpenAI ChatCompletion 接口，适用于如 gpt-3.5-turbo、gpt-4 等聊天模型。
    外部只需调用 self.generate() 或 self.chat_generate() 即可。
    """

	def __init__(self, model_name="gpt-3.5-turbo", gen_sentences=-1):
		# 使用新版本的 OpenAI 客户端
		self.client = openai.OpenAI(
			api_key=os.getenv("OPENAI_API_KEY", "sk-cucQ104SzP9t8h2zge5wRzsPUKZe8BD4GDz9AOJsGhuJHhdx"),
			base_url="https://xiaoai.plus/v1"
		)

		# 调整推理参数（与 ChatCompletion 相匹配）
		self.inference_args = {
			"model": model_name,  # gpt-3.5-turbo / gpt-4 etc.
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
			# 注意 ChatCompletion 下不再使用 "stop": "\n"
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences

	def generate(self, input_text: str, **_args):
		"""
        让外部直接调用 generate()，内部自动走到 chat_generate()。
        """
		messages = [
			{"role": "user", "content": input_text}
		]
		return self.chat_generate(messages, **_args)

	def _update_args(self, new_args):
		"""
        复用父类逻辑，但去除在 ChatCompletion 中不需要/不兼容的参数。
        """
		# 父类的 _update_args 会把 max_new_tokens -> max_tokens 等
		# 这里先把 stop/echo/return_full_text 等无用参数pop掉
		if "stop" in new_args:
			new_args.pop("stop")
		if "echo" in new_args:
			new_args.pop("echo")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		
		args = {**self.inference_args}
		from_cache = False
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")  # rely on caching
		if "num_return_sequences" in new_args:
			new_args["n"] = new_args.pop("num_return_sequences")
		if "repetition_penalty" in new_args:
			new_args["frequency_penalty"] = new_args.pop("repetition_penalty")
		return from_cache, {**args, **new_args}

	@lru_cache(maxsize=None)
	def _cached_generate(self, messages_tuple, **parameters):
		"""
        通过 ChatCompletion.create() 来做缓存调用
        """
		# 这里 messages 要能被hash，故转成 tuple
		messages = list(messages_tuple)
		response = self.client.chat.completions.create(messages=messages, **parameters)
		return response

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		"""
        真正的 ChatCompletion 调用：多轮对话（messages）结构。
        在这里可做缓存、多重重试等。
        """
		from_cache, parameters = self._update_args(gen_args)

		# 做一个可哈希版本的 messages，才能与 lru_cache 配合
		hashable_messages = [hashabledict(m) for m in messages]

		# 若 from_cache=True，则使用 _cached_generate 来存取缓存
		if from_cache:
			messages_tuple = tuple(hashable_messages)
			response = self._cached_generate(messages_tuple, **parameters)
		else:
			# 不从缓存，就直接调 API
			response = self.client.chat.completions.create(messages=messages, **parameters)

		# 将返回结果统一封装为 gen_output
		gen_output = []
		filtered_count = 0
		for i, resp in enumerate(response.choices):
			text = "" # 默认空字符串
			if resp.message and resp.message.content:
				text = resp.message.content
			else:
				# 检查 finish_reason
				finish_reason = resp.finish_reason
				if finish_reason == "content_filter":
					filtered_count += 1
					text = "" # 返回空字符串
				elif finish_reason == "length":
					text = resp.message.content or ""
				else:
					# 其他未知情况
					text = resp.message.content or ""

			# 如果有 gen_sentences 设置，并且 text 非空，则只取前几句
			if text and self.gen_sentences is not None:
				sentences = nltk.sent_tokenize(text)
				if len(sentences) > self.gen_sentences:
					text = " ".join(sentences[:self.gen_sentences])
			gen_output.append({"generated_text": text})

		return gen_output

	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		"""
        支持对多条对话并行生成（可选）。
        """
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]

class GPT35Turbo0613ChatModel(OpenAIChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="gpt-3.5-turbo-0613", gen_sentences=gen_sentences)

class DEPPSEEK(OpenAIChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="deepseek-v3", gen_sentences=gen_sentences)

class GPT4Turbo20240409ChatModel(OpenAIChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="gpt-4-turbo-2024-04-09", gen_sentences=gen_sentences)


class GPT4oMini20240718ChatModel(OpenAIChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="gpt-4o-mini-2024-07-18", gen_sentences=gen_sentences)

class GPT4oMinChatModel(OpenAIChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="gpt-4o-mini", gen_sentences=gen_sentences)

class AzureOpenAIModel(OpenAIModel):
	API_TOKEN = os.environ.get("MS_OPENAI_API_KEY")
	API_BASE = os.environ.get("MS_OPENAI_API_BASE")
	API_TYPE = "azure"
	API_VERSION = "2022-12-01"

	def __init__(self, model_name="chatgpt-turbo"):
		# 使用新版本的 Azure OpenAI 客户端
		from openai import AzureOpenAI
		self.client = AzureOpenAI(
			api_key=self.API_TOKEN,
			api_version=self.API_VERSION,
			azure_endpoint=self.API_BASE
		)
		
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
			"stop": "\n"
		}
		return


class AzureOpenAIChatModel(AzureOpenAIModel):
	def __init__(self, model_name="chatgpt", gen_sentences=-1):
		# 使用新版本的 Azure OpenAI 客户端
		from openai import AzureOpenAI
		self.client = AzureOpenAI(
			api_key=self.API_TOKEN,
			api_version="2023-03-15-preview",
			azure_endpoint=self.API_BASE
		)
		
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
			# "stop": "\n"  # no longer need since we are using chat
			# "echo": False,
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences
		return
	
	def _update_args(self, new_args):
		if "stop" in new_args:
			new_args.pop("stop")
		if "echo" in new_args:
			new_args.pop("echo")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		
		args = {**self.inference_args}
		from_cache = False
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")  # rely on caching
		if "num_return_sequences" in new_args:
			new_args["n"] = new_args.pop("num_return_sequences")
		if "repetition_penalty" in new_args:
			new_args["frequency_penalty"] = new_args.pop("repetition_penalty")
		return from_cache, {**args, **new_args}
	
	@lru_cache(maxsize=None)
	def _cached_generate(self, messages_tuple, **parameters):
		messages = list(messages_tuple)
		response = self.client.chat.completions.create(messages=messages, **parameters)
		return response
	
	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		# generate in a chat format
		from_cache, parameters = self._update_args(gen_args)
		hashable_messages = [hashabledict(m) for m in messages]
		if from_cache:
			messages_tuple = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
			response = self._cached_generate(messages_tuple, **parameters)
		else:
			response = self.client.chat.completions.create(messages=messages, **parameters)
		
		# format to a common format
		gen_output = []
		filtered_count = 0
		for i, resp in enumerate(response.choices):
			text = "" # 默认空字符串
			if resp.message and resp.message.content:
				text = resp.message.content
			else:
				# 检查 finish_reason
				finish_reason = resp.finish_reason
				if finish_reason == "content_filter":
					filtered_count += 1
					text = "" # 返回空字符串
				elif finish_reason == "length":
					text = resp.message.content or ""
				else:
					# 其他未知情况
					text = resp.message.content or ""

			# 如果有 gen_sentences 设置，并且 text 非空，则只取前几句
			if text and self.gen_sentences is not None:
				sentences = nltk.sent_tokenize(text)
				if len(sentences) > self.gen_sentences:
					text = " ".join(sentences[:self.gen_sentences])
			gen_output.append({"generated_text": text})

		return gen_output
	
	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]
	
	def generate(self, input_text, **_args):
		messages = [{
			"role": "user",
			"content": input_text
		}]
		return self.chat_generate(messages, **_args)


class LocalModel(GenerationModel):
	def __init__(self, model_name="EleutherAI/gpt-neo-2.7B", input_max_len=512, stop_symbol="\n", cuda=True):
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
		self.model = AutoModelForCausalLM.from_pretrained(model_name)
		stop_token_ids = self.tokenizer.encode(stop_symbol)[0]
		set_seed(42)
		if cuda and torch.cuda.is_available():
			self.cuda = True
			self.model = self.model.cuda()
		else:
			self.cuda = False
		
		self.input_max_len = input_max_len
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.7,
			"repetition_penalty": 1.0,
			"eos_token_id": stop_token_ids,
			"pad_token_id": self.tokenizer.eos_token_id
			# "return_full_text": False  # not available for manual generation
		}

	def generate(self, input_text:str, **gen_args):
		# override if gen_args specified
		gen_params = {**self.inference_args, **gen_args}
		inputs = self.tokenizer([input_text], return_tensors='pt', truncation=True, max_length=self.input_max_len)
		if self.cuda:
			inputs = {k: v.cuda() for k, v in inputs.items()}
		
		outputs = self.model.generate(**inputs, **gen_params)
		gen_only_outputs = outputs[:, len(inputs['input_ids'][0]):]
		gen_resps = self.tokenizer.batch_decode(gen_only_outputs, skip_special_tokens=True)

		# format output
		gen_output = []
		for resp in gen_resps:
			gen_output.append({"generated_text": resp})
		return gen_output


class DeepSeekChatModel(GenerationModel):
	"""
	DeepSeek Chat Model 基于 DeepSeek API，使用 OpenAI SDK 调用。
	支持 deepseek-chat 等模型。
	"""

	def __init__(self, model_name="deepseek-chat", gen_sentences=-1):
		# 设置 DeepSeek API 配置
		self.api_key = os.getenv("DEEPSEEK_API_KEY", "sk-d15a63e893c64c609a193c98e4fdee6f")
		
		# 使用 OpenAI SDK 连接 DeepSeek API
		self.client = openai.OpenAI(
			api_key=self.api_key,
			base_url="https://api.deepseek.com"
		)
		
		self.model_name = model_name
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"stream": False
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences

	def _update_args(self, new_args):
		"""
		更新生成参数，兼容不同的参数名称
		"""
		args = {**self.inference_args}
		from_cache = False
		
		# 参数名称转换
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")
		if "num_return_sequences" in new_args:
			# DeepSeek API 不支持多个返回序列，忽略此参数
			new_args.pop("num_return_sequences")
		if "n" in new_args:
			# DeepSeek API 只支持 n=1，强制设置为1
			new_args["n"] = 1
		if "repetition_penalty" in new_args:
			# DeepSeek API 使用不同的参数名，这里忽略
			new_args.pop("repetition_penalty")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		if "stop" in new_args:
			# 保留 stop 参数
			pass
		if "echo" in new_args:
			new_args.pop("echo")
			
		return from_cache, {**args, **new_args}

	@lru_cache(maxsize=None)
	def _cached_generate(self, messages_tuple, **parameters):
		"""
		缓存生成结果
		"""
		messages = list(messages_tuple)
		response = self.client.chat.completions.create(
			messages=messages,
			**parameters
		)
		return response

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		"""
		使用 DeepSeek API 生成聊天回复
		"""
		from_cache, parameters = self._update_args(gen_args)
		
		# 做一个可哈希版本的 messages，才能与 lru_cache 配合
		hashable_messages = [hashabledict(m) for m in messages]
		
		try:
			if from_cache:
				# 使用缓存
				messages_tuple = tuple(hashable_messages)
				response = self._cached_generate(messages_tuple, **parameters)
			else:
				# 直接调用 API
				response = self.client.chat.completions.create(
					messages=messages,
					**parameters
				)
			
			# 将返回结果统一封装
			gen_output = []
			for i, choice in enumerate(response.choices):
				text = ""
				if choice.message and choice.message.content:
					text = choice.message.content
				else:
					logger.error(f"DeepSeek API response choice {i} has no content. Full choice object: {choice}")
				
				# 如果有 gen_sentences 设置，并且 text 非空，则只取前几句
				if text and self.gen_sentences is not None:
					sentences = nltk.sent_tokenize(text)
					if len(sentences) > self.gen_sentences:
						text = " ".join(sentences[:self.gen_sentences])
				
				gen_output.append({"generated_text": text})
			
			return gen_output
			
		except Exception as e:
			logger.error(f"DeepSeek API 调用失败: {e}")
			raise

	def generate(self, input_text: str, **_args):
		"""
		简单的文本生成接口，内部转换为聊天格式
		"""
		messages = [
			{"role": "user", "content": input_text}
		]
		return self.chat_generate(messages, **_args)

	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		"""
		批量生成聊天回复
		"""
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]


class DashScopeChatModel(GenerationModel):
	"""
	阿里云百炼 Chat Model，基于 OpenAI 兼容接口调用。
	支持 qwen2-7b-instruct, qwen-plus, qwen-turbo 等模型。
	"""

	def __init__(self, model_name="qwen2-7b-instruct", gen_sentences=-1):
		# 设置阿里云百炼 API 配置

		
		# 使用 OpenAI SDK 连接阿里云百炼 API
		self.client = openai.OpenAI(
			api_key="sk-9762203f13924584a8abcff036725520",
			base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
		)
		
		self.model_name = model_name
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"stream": False
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences

	def _update_args(self, new_args):
		"""
		更新生成参数，兼容不同的参数名称
		"""
		args = {**self.inference_args}
		from_cache = False
		
		# 参数名称转换
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")
		if "num_return_sequences" in new_args:
			# 阿里云百炼 API 不支持多个返回序列，忽略此参数
			new_args.pop("num_return_sequences")
		if "n" in new_args:
			# 阿里云百炼 API 只支持 n=1，强制设置为1
			new_args["n"] = 1
		if "repetition_penalty" in new_args:
			# 阿里云百炼 API 使用不同的参数名，这里忽略
			new_args.pop("repetition_penalty")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		if "stop" in new_args:
			# 保留 stop 参数
			pass
		if "echo" in new_args:
			new_args.pop("echo")
		if "enable_thinking" in new_args:
			# 阿里云百炼 API 不支持 enable_thinking 参数，忽略
			new_args.pop("enable_thinking")
			
		return from_cache, {**args, **new_args}

	@lru_cache(maxsize=None)
	def _cached_generate(self, messages_tuple, extra_body_str="", **parameters):
		"""
		缓存生成结果
		"""
		messages = list(messages_tuple)
		if extra_body_str:
			import json
			extra_body = json.loads(extra_body_str)
			response = self.client.chat.completions.create(
				messages=messages,
				extra_body=extra_body,
				**parameters
			)
		else:
			response = self.client.chat.completions.create(
				messages=messages,
				**parameters
			)
		return response

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		"""
		使用阿里云百炼 API 生成聊天回复
		"""
		from_cache, parameters = self._update_args(gen_args)
		
		# 做一个可哈希版本的 messages，才能与 lru_cache 配合
		hashable_messages = [hashabledict(m) for m in messages]
		
		# 为某些模型添加 extra_body 参数
		extra_body = {}
		if self.model_name in ["qwen3-8b", "qwen-plus", "qwen-turbo", "qwen-max"]:
			extra_body = {"enable_thinking": False}
		
		try:
			if from_cache:
				# 使用缓存
				messages_tuple = tuple(hashable_messages)
				import json
				extra_body_str = json.dumps(extra_body) if extra_body else ""
				response = self._cached_generate(messages_tuple, extra_body_str=extra_body_str, **parameters)
			else:
				# 直接调用 API
				response = self.client.chat.completions.create(
					messages=messages,
					extra_body=extra_body,
					**parameters
				)
			
			# 将返回结果统一封装
			gen_output = []
			for i, choice in enumerate(response.choices):
				text = ""
				if choice.message and choice.message.content:
					text = choice.message.content
				else:
					logger.error(f"阿里云百炼 API response choice {i} has no content. Full choice object: {choice}")
				
				# 如果有 gen_sentences 设置，并且 text 非空，则只取前几句
				if text and self.gen_sentences is not None:
					sentences = nltk.sent_tokenize(text)
					if len(sentences) > self.gen_sentences:
						text = " ".join(sentences[:self.gen_sentences])
				
				gen_output.append({"generated_text": text})
			
			return gen_output
			
		except Exception as e:
			logger.error(f"阿里云百炼 API 调用失败: {e}")
			raise

	def generate(self, input_text: str, **_args):
		"""
		简单的文本生成接口，内部转换为聊天格式
		"""
		messages = [
			{"role": "user", "content": input_text}
		]
		return self.chat_generate(messages, **_args)

	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		"""
		批量生成聊天回复
		"""
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]


# 预定义的阿里云百炼模型类
class Qwen2_7B_InstructChatModel(DashScopeChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="qwen2-7b-instruct", gen_sentences=gen_sentences)


class QwenPlusChatModel(DashScopeChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="qwen2.5-7b-instruct", gen_sentences=gen_sentences)


class QwenTurboChatModel(DashScopeChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="qwen3-8b", gen_sentences=gen_sentences)


class QwenMaxChatModel(DashScopeChatModel):
	def __init__(self, gen_sentences=-1):
		super().__init__(model_name="qwen-max", gen_sentences=gen_sentences)


class LocalOpenAIChatModel(GenerationModel):
	"""
	本地 OpenAI 兼容模型，用于调用本地部署的模型服务
	支持通过本地 API 端点调用各种模型
	"""

	def __init__(self, model_name="xxx", base_url="http://localhost:6006/v1", gen_sentences=-1):
		# 使用新版本的 OpenAI 客户端连接本地服务
		self.client = openai.OpenAI(
			api_key="EMPTY",  # 本地服务通常不需要真实的 API key
			base_url=base_url
		)
		
		self.model_name = model_name
		self.base_url = base_url
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"stream": False  # 默认关闭流式输出，保持与其他模型一致
		}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences

	def _update_args(self, new_args):
		"""
		更新生成参数，兼容不同的参数名称
		"""
		args = {**self.inference_args}
		from_cache = False
		
		# 参数名称转换
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")
		if "num_return_sequences" in new_args:
			new_args["n"] = new_args.pop("num_return_sequences")
		if "repetition_penalty" in new_args:
			# 大多数本地模型服务不支持此参数，可以尝试转换或忽略
			new_args.pop("repetition_penalty")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		if "stop" in new_args:
			# 保留 stop 参数
			pass
		if "echo" in new_args:
			new_args.pop("echo")
			
		return from_cache, {**args, **new_args}

	@lru_cache(maxsize=None)
	def _cached_generate(self, messages_tuple, **parameters):
		"""
		缓存生成结果
		"""
		messages = list(messages_tuple)
		response = self.client.chat.completions.create(
			messages=messages,
			**parameters
		)
		return response

	def _fix_message_format(self, messages: List[Dict]) -> List[Dict]:
		"""
		修复消息格式，确保符合本地模型 u/a/u/a/u... 的要求
		"""
		if not messages:
			return messages
		
		# 首先收集所有system消息内容
		system_content = []
		non_system_messages = []
		
		for msg in messages:
			if msg["role"] == "system":
				system_content.append(msg["content"])
			else:
				non_system_messages.append(msg)
		
		if not non_system_messages:
			# 如果只有 system 消息，转换为一个 user 消息
			return [{"role": "user", "content": "\n".join(system_content)}]
		
		# 构建严格的 u/a/u/a/u... 交替模式
		fixed_messages = []
		
		# 第一个消息必须是 user
		first_msg = non_system_messages[0].copy()
		if system_content:
			# 将所有 system 内容合并到第一个消息中
			combined_content = "\n".join(system_content) + "\n" + first_msg["content"]
			first_msg = {"role": "user", "content": combined_content}
		else:
			first_msg = {"role": "user", "content": first_msg["content"]}
		
		fixed_messages.append(first_msg)
		
		# 处理剩余消息，严格交替 user/assistant
		expected_role = "assistant"  # 第一个消息是user，所以下一个应该是assistant
		
		for i, msg in enumerate(non_system_messages[1:], 1):
			new_msg = {"role": expected_role, "content": msg["content"]}
			fixed_messages.append(new_msg)
			# 切换期望的角色
			expected_role = "user" if expected_role == "assistant" else "assistant"
		
		# 确保最后一个消息是user（对话总是以user结束，等待assistant回复）
		if len(fixed_messages) > 1 and fixed_messages[-1]["role"] == "assistant":
			# 如果最后一个是assistant，我们需要调整
			# 移除最后一个assistant消息，确保以user结束
			if len(fixed_messages) % 2 == 0:  # 偶数个消息，应该以user结束
				# 将最后一个assistant消息的内容合并到前一个user消息中
				if len(fixed_messages) >= 2:
					last_assistant_content = fixed_messages[-1]["content"]
					fixed_messages[-2]["content"] += "\n" + last_assistant_content
					fixed_messages.pop()  # 移除最后一个assistant消息
		
		# 最终验证：确保格式正确
		if len(fixed_messages) > 0:
			# 必须以user开始
			if fixed_messages[0]["role"] != "user":
				fixed_messages[0]["role"] = "user"
			
			# 检查交替模式
			for i in range(1, len(fixed_messages)):
				expected = "assistant" if i % 2 == 1 else "user"
				if fixed_messages[i]["role"] != expected:
					fixed_messages[i]["role"] = expected
		
		return fixed_messages

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		"""
		使用本地 OpenAI 兼容 API 生成聊天回复
		"""
		from_cache, parameters = self._update_args(gen_args)
		
		# 修复消息格式以符合本地模型要求
		fixed_messages = self._fix_message_format(messages)
		
		# 做一个可哈希版本的 messages，才能与 lru_cache 配合
		hashable_messages = [hashabledict(m) for m in fixed_messages]
		
		try:
			if from_cache:
				# 使用缓存
				messages_tuple = tuple(hashable_messages)
				response = self._cached_generate(messages_tuple, **parameters)
			else:
				# 直接调用本地 API
				response = self.client.chat.completions.create(
					messages=fixed_messages,
					**parameters
				)
			
			# 将返回结果统一封装
			gen_output = []
			for i, choice in enumerate(response.choices):
				text = ""
				if choice.message and choice.message.content:
					text = choice.message.content
				else:
					logger.warning(f"本地模型 API response choice {i} has no content. Full choice object: {choice}")
				
				# 如果有 gen_sentences 设置，并且 text 非空，则只取前几句
				if text and self.gen_sentences is not None:
					sentences = nltk.sent_tokenize(text)
					if len(sentences) > self.gen_sentences:
						text = " ".join(sentences[:self.gen_sentences])
				
				gen_output.append({"generated_text": text})
			
			return gen_output
			
		except Exception as e:
			logger.error(f"本地模型 API 调用失败: {e}")
			logger.error(f"原始消息: {messages}")
			logger.error(f"修复后消息: {fixed_messages}")
			raise

	def generate(self, input_text: str, **_args):
		"""
		简单的文本生成接口，内部转换为聊天格式
		"""
		messages = [
			{"role": "user", "content": input_text}
		]
		return self.chat_generate(messages, **_args)

	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		"""
		批量生成聊天回复
		"""
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]

	def chat_with_stream(self, messages: List[Dict], **gen_args):
		"""
		支持流式输出的聊天生成方法（可选功能）
		这个方法可以用于需要实时输出的场景
		"""
		from_cache, parameters = self._update_args(gen_args)
		parameters["stream"] = True  # 启用流式输出
		
		# 修复消息格式以符合本地模型要求
		fixed_messages = self._fix_message_format(messages)
		
		try:
			response = self.client.chat.completions.create(
				messages=fixed_messages,
				**parameters
			)
			
			full_response = ""
			for chunk in response:
				if chunk.choices and len(chunk.choices) > 0:
					content = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
					if content:
						full_response += content
						yield content  # 实时输出每个token
			
			return full_response
			
		except Exception as e:
			logger.error(f"本地模型流式 API 调用失败: {e}")
			logger.error(f"原始消息: {messages}")
			logger.error(f"修复后消息: {fixed_messages}")
			raise


# 预定义的常用本地模型配置类
class LocalQwenChatModel(LocalOpenAIChatModel):
	"""本地部署的 Qwen 模型"""
	def __init__(self, gen_sentences=-1, base_url="http://localhost:6006/v1"):
		super().__init__(model_name="qwen", base_url=base_url, gen_sentences=gen_sentences)


class LocalLlamaChatModel(LocalOpenAIChatModel):
	"""本地部署的 Llama 模型"""
	def __init__(self, gen_sentences=-1, base_url="http://localhost:6006/v1"):
		super().__init__(model_name="llama", base_url=base_url, gen_sentences=gen_sentences)


class LocalChatGLMChatModel(LocalOpenAIChatModel):
	"""本地部署的 ChatGLM 模型"""
	def __init__(self, gen_sentences=-1, base_url="http://localhost:6006/v1"):
		super().__init__(model_name="chatglm", base_url=base_url, gen_sentences=gen_sentences)

