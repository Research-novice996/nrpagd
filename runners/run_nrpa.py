import numpy as np
import logging
import pickle
import argparse
import os
import time

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
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG

from core.nrpa import NRPAPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== NRPA 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

    system_name = PersuasionGame.SYS
    user_name = PersuasionGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

    game_ontology = PersuasionGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']

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
    elif cmd_args.llm == 'chatgpt':
        backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
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
    elif cmd_args.llm == 'deepseek-chat':
        backbone_model = DeepSeekChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = PersuaderChatModel
        UsrModel = PersuadeeChatModel
        SysPlanner = P4GChatSystemPlanner

    system = SysModel(
        sys_da,
        backbone_model,
        conv_examples=[exp_1],
        inference_args={
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        }
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

    with open(r"E:\PycharmProjects\NRPAGD-master\data\p4g\300_dialog_turn_based.pkl", "rb") as f:
        all_dialogs = pickle.load(f)

    num_dialogs = cmd_args.num_dialogs

    nrpa_args = dotdict({
        "nrpa_depth": cmd_args.nrpa_depth,
        "nrpa_iterations": cmd_args.reduced_iterations if cmd_args.reduced_iterations > 0 else cmd_args.nrpa_iterations,
        "nrpa_playout_epsilon": cmd_args.nrpa_playout_epsilon,
        "max_playout_steps": cmd_args.max_playout_steps,
        # 早停机制配置
        "early_stopping_enabled": cmd_args.early_stopping_enabled,
        "early_stopping_threshold": cmd_args.early_stopping_threshold,
        "early_stopping_patience": cmd_args.early_stopping_patience,
        "min_iterations": cmd_args.min_iterations,
    })
    print(f"NRPA 配置: 深度={nrpa_args.nrpa_depth}, 迭代次数={nrpa_args.nrpa_iterations}")
    if nrpa_args.early_stopping_enabled:
        print(f"早停机制: 启用, 阈值={nrpa_args.early_stopping_threshold}, 耐心值={nrpa_args.early_stopping_patience}, 最少迭代={nrpa_args.min_iterations}")

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

    bad_dialogs = [
        '20180808-024552_152_live',
        '20180723-100140_767_live',
        '20180825-080802_964_live'
    ]

    dialog_keys_to_process = [k for k in all_dialogs.keys() if k not in bad_dialogs]
    target_dialogs_count = min(num_dialogs, len(dialog_keys_to_process))

    needed_new_dialogs = target_dialogs_count - len(processed_dialogs)

    if needed_new_dialogs <= 0:
        print(f"已完成所有 {target_dialogs_count} 个目标对话的处理")
        return

    print(f"目标处理 {target_dialogs_count} 个对话，当前已处理 {len(processed_dialogs)} 个，将处理 {needed_new_dialogs} 个新对话")
    num_done = 0
    pbar = tqdm(total=needed_new_dialogs, desc="Evaluating")

    dialog_count = 0  # 添加对话计数器
    for did in dialog_keys_to_process:
        dialog_count += 1  # 增加计数器
        
        # 如果指定了起始对话编号，跳过前面的对话
        if hasattr(cmd_args, 'start_dialog') and dialog_count < cmd_args.start_dialog:
            print(f"跳过对话 {dialog_count}: {did}")
            continue
            
        if did in processed_dialogs:
            continue

        if num_done >= needed_new_dialogs:
            print(f"已完成 {needed_new_dialogs} 个新对话的处理目标")
            break

        print(f"\n评估对话ID: {did} (对话编号: {dialog_count}, 新对话 {num_done+1}/{needed_new_dialogs})")
        context = ""
        dialog = all_dialogs[did]
        state = game.init_dialog()

        turn_successful = True
        for t, turn in enumerate(dialog["dialog"]):
            if len(turn["ee"]) == 0:
                print(f"对话 {did} 在轮次 {t} 因用户无响应而结束")
                break

            if t == len(dialog["dialog"]) - 1:
                print(f"对话 {did} 到达最后一轮，停止处理")
                break

            usr_utt = " ".join(turn["ee"]).strip()
            usr_da = dialog["label"][t]["ee"][-1] if len(dialog["label"][t]["ee"]) > 0 else "neutral"

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

            sys_utt = " ".join(turn["er"]).strip()
            sys_da_set = set(dialog["label"][t]["er"])
            intersected_das = sys_da_set.intersection(system.dialog_acts)
            sys_da_final = list(intersected_das)[-1] if intersected_das else "other"

            state.add_single(PersuasionGame.SYS, sys_da_final, sys_utt)
            state.add_single(PersuasionGame.USR, usr_da, usr_utt)

            if usr_da == PersuasionGame.U_Donate:
                print(f"对话 {did} 在轮次 {t} 用户同意捐赠，结束处理此对话")
                break
            context = f"""
            {context}
            {PersuasionGame.SYS}: {sys_utt}
            {PersuasionGame.USR}: {usr_utt}
            """.replace('\t', '').strip()

            if hasattr(backbone_model, '_cached_generate'):
                backbone_model._cached_generate.cache_clear()

            # 记录调用 nrpa 前的 state 的历史长度
            len_history_before_nrpa_call = len(state.history)

            # 记录NRPA搜索时间
            nrpa_start_time = time.time()
            dialog_planner = NRPAPlanner(game, planner, nrpa_args)
            final_state = dialog_planner.nrpa(nrpa_args.nrpa_depth, {}, state.copy())
            nrpa_end_time = time.time()
            nrpa_duration = nrpa_end_time - nrpa_start_time
            print(f"NRPA搜索耗时: {nrpa_duration:.2f}秒")

            nrpa_policy_next_da = "other"
            nrpa_pred_rep = "Could not determine a response."
            next_da_idx = -1

            if final_state and len(final_state.history) > len(state.history):
                # 首先检查 final_state 是否已经是终止状态
                if dialog_planner.terminal(final_state):
                    print(f"注意：NRPA 返回的是终止状态（对话已结束）")
                    # 检查终止原因
                    terminal_value = game.get_dialog_ended(final_state)
                    if terminal_value == 1.0:
                        print(f"  终止原因：用户同意捐赠")
                    elif terminal_value == -1.0:
                        print(f"  终止原因：用户拒绝捐赠或对话轮次超限")
                    
                    # 即使是终止状态，也要从中提取系统的最后回复
                    current_state_len = len(state.history)
                    for i in range(current_state_len, len(final_state.history)):
                        next_turn = final_state.history[i]
                        speaker, da, utterance = next_turn
                        if speaker == PersuasionGame.SYS:
                            nrpa_policy_next_da = da
                            nrpa_pred_rep = utterance
                            print(f"DEBUG new_resp (NRPA Path, Terminal): Speaker: {speaker}, Utterance for new_resp: {utterance}")
                            break
                        else:
                            print(f"DEBUG new_resp (NRPA Path, Terminal, skipped): Speaker: {speaker}, Utterance: {utterance}")
                else:
                    print(f"NRPA 返回非终止状态，正常提取下一步动作")
                    # 只有在非终止状态下才提取动作
                    current_state_len = len(state.history)
                    for i in range(current_state_len, len(final_state.history)):
                        next_turn = final_state.history[i]
                        speaker, da, utterance = next_turn
                        if speaker == PersuasionGame.SYS:
                            nrpa_policy_next_da = da
                            nrpa_pred_rep = utterance
                            print(f"DEBUG new_resp (NRPA Path): Speaker: {speaker}, Utterance for new_resp: {utterance}")
                            break
                        else:
                            print(f"DEBUG new_resp (NRPA Path, skipped): Speaker: {speaker}, Utterance: {utterance}")

            else:
                # 详细诊断信息
                print("警告：NRPA未能找到有效的下一步路径，将随机选择")
                print(f"  输入状态轮次: {len(state.history)}")
                if final_state is None:
                    print(f"  NRPA返回: None")
                else:
                    print(f"  NRPA返回状态轮次: {len(final_state.history)}")
                    print(f"  是否终止状态: {dialog_planner.terminal(final_state)}")
                print(f"  当前对话轮次: t={t}")
                legal_moves = dialog_planner.legalMoves(state)
                if legal_moves:
                    next_da_idx = np.random.choice(legal_moves)
                    nrpa_policy_next_da = system.dialog_acts[next_da_idx]
                    temp_next_state = game.get_next_state(state.copy(), next_da_idx)
                    if len(temp_next_state.history) > len(state.history):
                        # game.get_next_state 生成两轮：系统回复 + 用户回复
                        # 我们需要的是系统回复，不是用户回复
                        for i in range(len(state.history), len(temp_next_state.history)):
                            turn = temp_next_state.history[i]
                            speaker, da, utterance = turn
                            if speaker == PersuasionGame.SYS:
                                nrpa_pred_rep = utterance
                                print(f"DEBUG new_resp (Fallback Path): Speaker: {speaker}, Utterance for new_resp: {utterance}")
                                break
                        else:
                            nrpa_pred_rep = "Error: Could not find system response"
                            print(f"DEBUG new_resp (Fallback Path): Error: Could not find system response in temp_state")
                    else:
                        nrpa_pred_rep = "Error generating random response"
                        print(f"DEBUG new_resp (Fallback Path): Error generating random response, state history did not grow.")

                else:
                    print(f"错误：对话 {did} 轮次 {t} 无合法动作可选！跳过此轮次")
                    turn_successful = False
                    if 'nrpa_pred_rep' not in locals():
                         nrpa_pred_rep = "No legal moves, no response generated."
                    break

            human_resp = "No human response available (last turn)."
            next_sys_da = "N/A"

            if (t + 1) < len(dialog["dialog"]):
                # --- 诊断信息打印开始 ---
                next_turn_data = dialog["dialog"][t + 1]
                print(f"DEBUG: Dialog ID {did}, Current Turn t={t}, Next Turn t+1 Data:")
                print(f"  next_turn_data['er'] (expected Persuader): {next_turn_data.get('er')}")
                print(f"  next_turn_data['ee'] (expected Persuadee): {next_turn_data.get('ee')}")
                # --- 诊断信息打印结束 ---

                # Attempt to get Persuader's response from the next turn ("er" field)
                if "er" in dialog["dialog"][t + 1] and isinstance(dialog["dialog"][t + 1]["er"], list):
                    human_resp = " ".join(dialog["dialog"][t + 1]["er"]).strip()
                    if not human_resp and dialog["dialog"][t + 1]["er"] is not None: # Check if "er" was present but empty list
                        human_resp = "Persuader response was empty in data."
                else:
                    print(f"警告: 对话 {did} 轮次 {t+1} 的 'er' 字段（说服者发言）缺失或格式不正确。")
                    human_resp = "Persuader response data missing/malformed."

                # Attempt to get Persuader's dialog act from the next turn labels ("er" field)
                if "er" in dialog["label"][t + 1] and isinstance(dialog["label"][t + 1]["er"], list):
                    next_sys_das = set(dialog["label"][t + 1]["er"])
                    if not next_sys_das and dialog["label"][t + 1]["er"] is not None: # Check if "er" was present but empty list
                        next_sys_da = "Persuader DA was empty in data."
                    else:
                        next_intersected_das = next_sys_das.intersection(system.dialog_acts)
                        next_sys_da = list(next_intersected_das)[-1] if next_intersected_das else "other"
                else:
                    print(f"警告: 对话 {did} 轮次 {t+1} 标签的 'er' 字段（说服者对话行为）缺失或格式不正确。")
                    next_sys_da = "Persuader DA data missing/malformed."

            debug_data = {
                "selected_da_idx": next_da_idx,
                "selected_da": nrpa_policy_next_da,
                "final_state_score": dialog_planner.score(final_state, len_history_before_nrpa_call) if final_state else None,
                "final_state_history_len": len(final_state.history) if final_state else 0,
                "nrpa_iterations": nrpa_args.nrpa_iterations,
                "nrpa_depth": nrpa_args.nrpa_depth,
                "nrpa_search_time": nrpa_duration,
            }

            cmp_data = {
                'did': did,
                'turn': t,
                'context': context,
                'ori_resp': human_resp,
                'ori_da': next_sys_da,
                'new_resp': nrpa_pred_rep,
                'new_da': nrpa_policy_next_da,
                'debug': debug_data
            }
            output.append(cmp_data)

        if not turn_successful:
             print(f"对话 {did} 处理中遇到错误，可能未完整记录")

        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)
            print(f"对话 {did} 处理完毕，结果已保存。当前总记录数: {len(output)}")

        processed_dialogs.add(did)
        num_done += 1
        pbar.update(1)

    pbar.close()
    
    # 计算并显示总运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== NRPA 实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))} ===")
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟) ===")
    print(f"=== 平均每个对话处理时间: {total_duration/len(processed_dialogs):.2f}秒 ===")
    
    print(f"\n所有对话处理完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数: {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default=r"D:\GDPZero-master\outputs\gpt-4o-mini-2024-07-18_nrpa3_10sims_20dialogs.pkl",
                        help='output file path')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "gpt-3.5-turbo-0613", "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat"],
                        help='LLM backbone model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences for LLM generation')
    parser.add_argument('--nrpa_depth', type=int, default= 3, help='NRPA recursive search depth')
    parser.add_argument('--nrpa_iterations', type=int, default=10, help='Number of iterations per NRPA level')
    parser.add_argument('--num_dialogs', type=int, default=1, help='Target number of dialogs to process')
    parser.add_argument('--nrpa_playout_epsilon', type=float, default=0, help='Epsilon for epsilon-greedy exploration in NRPA playouts')
    parser.add_argument('--reduced_iterations', type=int, default=0, help='Reduced number of iterations (overrides nrpa_iterations if > 0)')
    parser.add_argument('--max_playout_steps', type=int, default=0, help='Maximum playout steps (0 means unlimited)')
    parser.add_argument('--start_dialog', type=int, default=1, help='Start processing from dialog number (1-based)')
    parser.add_argument('--early_stopping_enabled', type=bool, default=True, help='Enable early stopping mechanism')
    parser.add_argument('--early_stopping_threshold', type=float, default=1.003, help='Early stopping threshold')
    parser.add_argument('--early_stopping_patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--min_iterations', type=int, default=100, help='Minimum iterations')

    cmd_args = parser.parse_args()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    print("命令行参数:")
    print(f"  Output file: {cmd_args.output}")
    print(f"  LLM Model: {cmd_args.llm}")
    print(f"  Max Gen Sentences: {cmd_args.gen_sentences}")
    print(f"  NRPA Depth: {cmd_args.nrpa_depth}")
    print(f"  NRPA Iterations: {cmd_args.nrpa_iterations}")
    print(f"  Num Dialogs: {cmd_args.num_dialogs}")
    print(f"  NRPA Playout Epsilon: {cmd_args.nrpa_playout_epsilon}")
    print(f"  Start Dialog: {cmd_args.start_dialog}")
    if cmd_args.reduced_iterations > 0:
        print(f"  Using Reduced Iterations: {cmd_args.reduced_iterations}")
    if cmd_args.max_playout_steps > 0:
        print(f"  Max Playout Steps: {cmd_args.max_playout_steps}")
    if cmd_args.early_stopping_enabled:
        print(f"  Early Stopping: Enabled, Threshold={cmd_args.early_stopping_threshold}, Patience={cmd_args.early_stopping_patience}, Minimum Iterations={cmd_args.min_iterations}")

    main(cmd_args)
