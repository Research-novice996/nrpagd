import numpy as np
import logging
import pickle
import argparse
import os

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, GPT35Turbo0613ChatModel, GPT4Turbo20240409ChatModel, GPT4oMini20240718ChatModel, DeepSeekChatModel
)
from core.players import (
	PersuadeeModel, PersuaderModel, P4GSystemPlanner,
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame
from core.helpers import DialogSession
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	if cmd_args.llm == 'code-davinci-002':
		backbone_model = OpenAIModel(cmd_args.llm)
		SysModel = PersuaderModel
		UsrModel = PersuadeeModel
		SysPlanner = P4GSystemPlanner
	elif cmd_args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'gpt-3.5-turbo-0613':
		backbone_model = GPT35Turbo0613ChatModel(cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'gpt-4-turbo-2024-04-09':
		backbone_model = GPT4Turbo20240409ChatModel(cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'gpt-4o-mini-2024-07-18':
		backbone_model = GPT4oMini20240718ChatModel(cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'deepseek-chat':
		backbone_model = DeepSeekChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner

	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']

	system = SysModel(
		sys_da,
		backbone_model,
		conv_examples=[exp_1]
	)
	user = UsrModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"return_full_text": False,
		},
		backbone_model=backbone_model, 
		conv_examples=[exp_1]
	)
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)
	game = PersuasionGame(system, user)

	with open(r"C:\Users\Windows11\Desktop\GDPZero-master\data\p4g\300_dialog_turn_based.pkl", "rb") as f:
		all_dialogs = pickle.load(f)

	output = []
	processed_dialogs = set()
	start_from_scratch = False

	if os.path.exists(cmd_args.output):
		try:
			with open(cmd_args.output, "rb") as f:
				output = pickle.load(f)
				print(f"已加载现有输出文件，包含 {len(output)} 条记录")

				for item in output:
					if 'did' in item:
						processed_dialogs.add(item['did'])
				print(f"已处理的对话ID ({len(processed_dialogs)} 个): {processed_dialogs}")
		except Exception as e:
			print(f"读取输出文件失败: {e}")
			print("将创建新的输出文件")
			start_from_scratch = True
	else:
		print(f"输出文件 {cmd_args.output} 不存在，将创建新文件")
		start_from_scratch = True

	bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']

	dialog_keys_to_process = [k for k in all_dialogs.keys() if k not in bad_dialogs]
	target_dialogs_count = min(cmd_args.num_dialogs, len(dialog_keys_to_process))

	needed_new_dialogs = target_dialogs_count - len(processed_dialogs)

	if needed_new_dialogs <= 0:
		print(f"已完成所有 {target_dialogs_count} 个目标对话的处理")
		return

	print(f"目标处理 {target_dialogs_count} 个对话，当前已处理 {len(processed_dialogs)} 个，将处理 {needed_new_dialogs} 个新对话")
	num_done = 0
	pbar = tqdm(total=needed_new_dialogs, desc="evaluating")

	for did in dialog_keys_to_process:
		if did in processed_dialogs:
			continue

		if num_done >= needed_new_dialogs:
			print(f"已完成 {needed_new_dialogs} 个新对话的处理目标")
			break

		print(f"\n评估对话ID: {did} (新对话 {num_done+1}/{needed_new_dialogs})")
		context = ""
		no_error = True
		dialog = all_dialogs[did]
		
		state = game.init_dialog()
		turn_successful = True
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:  # ended
				print(f"对话 {did} 在轮次 {t} 因用户无响应而结束")
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				print(f"对话 {did} 到达最后一轮，停止处理")
				break

			usr_utt = " ".join(turn["ee"]).strip()
			usr_da = dialog["label"][t]["ee"][-1] if len(dialog["label"][t]["ee"]) > 0 else "neutral"

			# map to our dialog act
			if usr_da == "disagree-donation":
				usr_da = PersuasionGame.U_NoDonation
			elif usr_da == "negative-reaction-to-donation":
				usr_da = PersuasionGame.U_NegativeReaction
			elif usr_da == "positive-reaction-to-donation":
				usr_da = PersuasionGame.U_PositiveReaction
			elif usr_da == "agree-donation":
				usr_da = PersuasionGame.U_Donate
			else:
				usr_da = PersuasionGame.U_Neutral

			# game ended
			if usr_da == PersuasionGame.U_Donate:
				break

			# map sys as well
			sys_utt = " ".join(turn["er"]).strip()
			sys_da = set(dialog["label"][t]["er"])
			intersected_das = sys_da.intersection(system.dialog_acts)
			if len(intersected_das) == 0:
				sys_da = "other"
			else:
				sys_da = list(intersected_das)[-1]
			
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			# update context for evaluation
			context = f"""
			{context}
			Persuader: {sys_utt}
			Persuadee: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# mcts policy
			prior, v = planner.predict(state)
			greedy_policy = system.dialog_acts[np.argmax(prior)]
			try:
				next_best_state = game.get_next_state(state, np.argmax(prior))
			except Exception as e:
				bad_dialogs.append(did)
				no_error = False
				raise e
			greedy_pred_resp = next_best_state.history[-2][2]

			human_resp = "No human response available (last turn)."
			next_sys_da = "N/A"

			if (t + 1) < len(dialog["dialog"]):
				# Attempt to get Persuader's response from the next turn ("er" field)
				if "er" in dialog["dialog"][t + 1] and isinstance(dialog["dialog"][t + 1]["er"], list):
					human_resp = " ".join(dialog["dialog"][t + 1]["er"]).strip()
					if not human_resp and dialog["dialog"][t + 1]["er"] is not None:
						human_resp = "Persuader response was empty in data."
				else:
					print(f"警告: 对话 {did} 轮次 {t+1} 的 'er' 字段（说服者发言）缺失或格式不正确。")
					human_resp = "Persuader response data missing/malformed."

				# Attempt to get Persuader's dialog act from the next turn labels ("er" field)
				if "er" in dialog["label"][t + 1] and isinstance(dialog["label"][t + 1]["er"], list):
					next_sys_das = set(dialog["label"][t+1]["er"])
					if not next_sys_das and dialog["label"][t + 1]["er"] is not None:
						next_sys_da = "Persuader DA was empty in data."
					else:
						next_intersected_das = next_sys_das.intersection(system.dialog_acts)
						next_sys_da = list(next_intersected_das)[-1] if next_intersected_das else "other"
				else:
					print(f"警告: 对话 {did} 轮次 {t+1} 标签的 'er' 字段（说服者对话行为）缺失或格式不正确。")
					next_sys_da = "Persuader DA data missing/malformed."

			# logging for debug
			debug_data = {
				"prior": prior,
				"da": greedy_policy,
				"v": v
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': greedy_pred_resp,
				'new_da': greedy_policy,
				"debug": debug_data,
			}
			output.append(cmp_data)
		
		if not no_error:
			print(f"对话 {did} 处理中遇到错误，可能未完整记录")

		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
			print(f"对话 {did} 处理完毕，结果已保存。当前总记录数: {len(output)}")

		processed_dialogs.add(did)
		num_done += 1
		pbar.update(1)
	pbar.close()
	print(f"\n所有对话处理完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数: {len(output)}")
	with open(cmd_args.output, "wb") as f:
		pickle.dump(output, f)
	print(f"最终结果已保存到: {cmd_args.output}")
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18", choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "gpt-3.5-turbo-0613", "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences to generate. -1 for no limit')
	parser.add_argument('--output', type=str, default="C:\\\\Users\\\\Windows11\\\\Desktop\\\\GDPZero-master\\\\outputs\\\\gpt-4o-mini-2024-07-18_raw_prompt_20dialogs.pkl", help='output file')
	parser.add_argument('--num_dialogs', type=int, default=20, help='Target number of dialogs to process')
	cmd_args = parser.parse_args()
	print("saving to", cmd_args.output)
	print(f"LLM Model: {cmd_args.llm}")
	print(f"Max Gen Sentences: {cmd_args.gen_sentences}")
	print(f"Target Dialogs: {cmd_args.num_dialogs}")

	main(cmd_args)