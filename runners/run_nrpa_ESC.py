import numpy as np
import logging
import pickle
import argparse
import os
import time

from tqdm.auto import tqdm

from core.gen_models import (
    LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, GPT35Turbo0613ChatModel, GPT4Turbo20240409ChatModel,
    GPT4oMini20240718ChatModel, DeepSeekChatModel, DashScopeChatModel,
    Qwen2_7B_InstructChatModel, QwenPlusChatModel, QwenTurboChatModel, QwenMaxChatModel
)
from core.players import (
    PersuadeeModel, PersuaderModel, P4GSystemPlanner,
    PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.esc_players import (
	TherapistModel, PatientModel, ESCSystemPlanner,
	TherapistChatModel, PatientChatModel, ESCChatSystemPlanner
)
from core.cb_players import (
	BuyerModel, SellerModel, CBSystemPlanner,
	BuyerChatModel, SellerChatModel, CBChatSystemPlanner
)
from core.game1 import EmotionalSupportGame
from core.game import PersuasionGame
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import ESConv_EXP_DIALOG, CB_EXP_DIALOG

from core.nrpa_ESC import NRPAPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== NRPA 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

    system_name = EmotionalSupportGame.SYS
    user_name = EmotionalSupportGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(ESConv_EXP_DIALOG)

    game_ontology = EmotionalSupportGame.get_game_ontology()
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
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'deepseek-chat':
        backbone_model = DeepSeekChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = PersuaderChatModel
        UsrModel = PersuadeeChatModel
        SysPlanner = P4GChatSystemPlanner
    elif cmd_args.llm == 'qwen2-7b-instruct':
        backbone_model = Qwen2_7B_InstructChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'qwen-plus':
        backbone_model = QwenPlusChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'qwen-turbo':
        backbone_model = QwenTurboChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'qwen-max':
        backbone_model = QwenMaxChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    else:
        raise ValueError(f"不支持的模型: {cmd_args.llm}")

    system = SysModel(
        sys_da,
        backbone_model,
        conv_examples=[exp_1],
        inference_args={
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        },
        zero_shot=False  # 确保与游戏设置一致
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
        conv_examples=[exp_1],
        zero_shot=False  # 确保与游戏设置一致
    )

    planner = SysPlanner(
        dialog_acts=system.dialog_acts,
        max_hist_num_turns=system.max_hist_num_turns,
        user_dialog_acts=user.dialog_acts,
        user_max_hist_num_turns=user.max_hist_num_turns,
        generation_model=backbone_model,
        conv_examples=[exp_1],
        zero_shot=False  # 确保与游戏设置一致
    )

    game = EmotionalSupportGame(system, user, planner, zero_shot=False)
    print(f"使用模型: {cmd_args.llm}")
    print(f"系统对话行为: {system.dialog_acts}")
    print(f"用户对话行为: {user.dialog_acts}")

    import json
    all_dialogs = {}
    with open(r"C:\Users\Windows11\Desktop\GDPZero-master\data\esc-train.txt", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:  # 跳过空行
                try:
                    dialog_data = json.loads(line)
                    # 使用行号作为对话ID，或者如果有其他唯一标识符可以使用那个
                    dialog_id = f"dialog_{line_num}"
                    all_dialogs[dialog_id] = dialog_data
                except json.JSONDecodeError as e:
                    print(f"跳过第 {line_num} 行，JSON解析错误: {e}")
                    continue

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
    # print(f"NRPA 配置: 深度={nrpa_args.nrpa_depth}, 迭代次数={nrpa_args.nrpa_iterations}")
    # if nrpa_args.early_stopping_enabled:
    #     print(
    #         f"早停机制: 启用, 阈值={nrpa_args.early_stopping_threshold}, 耐心值={nrpa_args.early_stopping_patience}, 最少迭代={nrpa_args.min_iterations}")

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

    print(
        f"目标处理 {target_dialogs_count} 个对话，当前已处理 {len(processed_dialogs)} 个，将处理 {needed_new_dialogs} 个新对话")
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

        # print(f"\n=== 评估对话ID: {did} (对话编号: {dialog_count}) ===")
        dialog = all_dialogs[did]

        # 从对话数据中提取情感类型和问题类型，如果没有则使用默认值
        emotion_type = dialog.get("emotion_type", "unknown")
        problem_type = dialog.get("problem_type", "unknown")

        # print(f"情感类型: {emotion_type}, 问题类型: {problem_type}")

        # 获取situation信息
        situation = dialog.get("situation", "")
        # print(f"情况描述: {situation}")

        state = game.init_dialog(emotion_type, problem_type)

        # 构建基础context（包含situation对话）
        base_context = ""
        if situation:
            usr_utt_init = "Hello. " + situation
            sys_utt_init = "Hello!"

            base_context = f"Therapist: {sys_utt_init}\nPatient: {usr_utt_init}"

            # 添加初始的situation对话到state
            usr_da = EmotionalSupportGame.U_FeelTheSame
            sys_da = EmotionalSupportGame.S_Others
            state.add_single(EmotionalSupportGame.SYS, sys_da, sys_utt_init)
            state.add_single(EmotionalSupportGame.USR, usr_da, usr_utt_init)

        dialog_turns = dialog["dialog"]

        # 找到用户和系统的轮次对
        usr_turns = []
        sys_turns = []

        for turn in dialog_turns:
            if turn["speaker"] == "usr":
                usr_turns.append(turn)
            elif turn["speaker"] == "sys":
                sys_turns.append(turn)

        # 确保我们有足够的轮次来进行对话
        max_pairs = min(len(usr_turns), len(sys_turns))
        if max_pairs == 0:
            # print(f"对话 {did} 没有足够的轮次")
            continue

        turn_successful = True
        # 逐轮处理对话，使用PatientChatModel生成正确的对话行为
        for t in range(max_pairs - 1):  # Skip last turn
            # print(f"\n=== 第 {t+1} 轮对话开始 ===")

            # 构建当前轮次的状态（基于原始数据集，不受NRPA影响）
            current_state = game.init_dialog(emotion_type, problem_type)

            # 添加初始的situation对话（如果存在）
            if situation:
                usr_utt_init = "Hello. " + situation
                usr_da_init = EmotionalSupportGame.U_FeelTheSame
                sys_utt_init = "Hello!"
                sys_da_init = EmotionalSupportGame.S_Others
                current_state.add_single(EmotionalSupportGame.SYS, sys_da_init, sys_utt_init)
                current_state.add_single(EmotionalSupportGame.USR, usr_da_init, usr_utt_init)

            # 添加数据集中的历史对话（跳过前两句，因为已经被situation替换）
            for i in range(t):  # 添加到当前轮次之前的对话
                # 数据集索引需要+1，因为跳过了前两句（被situation替换）
                dataset_idx = i + 1

                if dataset_idx < len(sys_turns):
                    # 添加系统回应
                    sys_utt_i = sys_turns[dataset_idx]["text"].strip()
                    sys_strategy_i = sys_turns[dataset_idx].get("strategy", "Others")
                    sys_da_i = sys_strategy_i if sys_strategy_i in system.dialog_acts else "Others"
                    current_state.add_single(EmotionalSupportGame.SYS, sys_da_i, sys_utt_i)

                if dataset_idx < len(usr_turns):
                    # 添加用户回应，使用PatientChatModel生成DA
                    usr_utt_i = usr_turns[dataset_idx]["text"].strip()

                    # 使用PatientChatModel生成用户对话行为 (当zero_shot=False时)
                    if not game.zero_shot:
                        try:
                            temp_state = current_state.copy()
                            usr_da_i, generated_response = user.get_utterance_w_da(temp_state, None, mode='train')
                        except Exception as e:
                            # print(f"  警告: 使用PatientChatModel生成用户DA失败: {e}")
                            usr_da_i = EmotionalSupportGame.U_FeelTheSame
                    else:
                        # zero_shot模式下，根据情感类型映射用户对话行为
                        emotion_type_temp = dialog.get("emotion_type", "")
                        if emotion_type_temp in ["depression", "sadness"]:
                            usr_da_i = EmotionalSupportGame.U_FeelWorse
                        elif emotion_type_temp in ["anxiety", "anger"]:
                            usr_da_i = EmotionalSupportGame.U_FeelTheSame
                        else:
                            usr_da_i = EmotionalSupportGame.U_FeelBetter

                    current_state.add_single(EmotionalSupportGame.USR, usr_da_i, usr_utt_i)

            # 显示当前轮次开始时的对话状态
            print(f"当前对话状态 (第 {t+1} 轮开始前):")
            if len(current_state.history) == 0:
                print("  [空对话状态]")
            else:
                for i, (role, da, utt) in enumerate(current_state.history):
                    print(f"  [{i}] {role}: [{da}] {utt}")

            if t >= len(usr_turns):
                # print(f"对话 {did} 在第 {t} 轮结束，没有更多用户回应")
                break

            # 获取当前轮次的用户和系统话语（跳过前两句，索引+1）
            dataset_idx = t + 1  # 跳过被situation替换的前两句

            if dataset_idx < len(usr_turns):
                usr_utt = usr_turns[dataset_idx]["text"].strip()
            else:
                usr_utt = "Thank you."

            if dataset_idx < len(sys_turns):
                sys_utt = sys_turns[dataset_idx]["text"].strip()
                sys_strategy = sys_turns[dataset_idx].get("strategy", "Others")
                sys_da_final = sys_strategy if sys_strategy in system.dialog_acts else "Others"
            else:
                sys_utt = "Hello!"
                sys_da_final = "Others"

            # 获取用户DA（从已构建的状态中）
            if len(current_state.history) > 0 and current_state.history[-1][0] == EmotionalSupportGame.USR:
                usr_da = current_state.history[-1][1]
            else:
                usr_da = EmotionalSupportGame.U_FeelTheSame

            # 基于current_state构建完整的context
            context = ""
            for role, da, utt in current_state.history:
                if role == EmotionalSupportGame.SYS:
                    context += f"Therapist: {utt}\n"
                else:
                    context += f"Patient: {utt}\n"
            context = context.strip()

            if hasattr(backbone_model, '_cached_generate'):
                backbone_model._cached_generate.cache_clear()

            # 记录调用 nrpa 前的 current_state 的历史长度
            len_history_before_nrpa_call = len(current_state.history)

            # NRPA搜索
            nrpa_start_time = time.time()
            dialog_planner = NRPAPlanner(game, planner, nrpa_args)
            final_state = dialog_planner.nrpa(nrpa_args.nrpa_depth, {}, current_state.copy())
            nrpa_end_time = time.time()
            nrpa_duration = nrpa_end_time - nrpa_start_time

            # # 显示NRPA模拟完成后的完整对话
            # if final_state and len(final_state.history) > len(current_state.history):
            #     print(f"\nNRPA模拟完成后的完整对话:")
            #     for i, (role, da, utt) in enumerate(final_state.history):
            #         if i < len(current_state.history):
            #             print(f"  [{i}] {role}: [{da}] {utt}")
            #         else:
            #             print(f"  [{i}] {role}: [{da}] {utt} ← 新生成")
            #     print("")
            # elif final_state:
            #     print(f"NRPA返回状态长度未增长: {len(final_state.history)}")
            # else:
            #     print(f"NRPA返回None")

            nrpa_policy_next_da = "other"
            nrpa_pred_rep = "Could not determine a response."
            next_da_idx = -1

            if final_state and len(final_state.history) > len(current_state.history):
                # 首先检查 final_state 是否已经是终止状态
                if dialog_planner.terminal(final_state):
                    print(f"注意：NRPA 返回的是终止状态（对话已结束）")
                    # 检查终止原因
                    terminal_value = game.get_dialog_ended(final_state)
                    if terminal_value == 1.0:
                         print(f"  终止原因：问题已解决")
                    elif terminal_value == -1.0:
                         print(f"  终止原因：问题未解决或对话轮次超限")

                    # 即使是终止状态，也要从中提取系统的最后回复
                    current_state_len = len(current_state.history)
                    for i in range(current_state_len, len(final_state.history)):
                        next_turn = final_state.history[i]
                        speaker, da, utterance = next_turn
                        if speaker == EmotionalSupportGame.SYS:
                            nrpa_policy_next_da = da
                            nrpa_pred_rep = utterance
                            print(f"DEBUG new_resp (NRPA Path, Terminal): Speaker: {speaker}, Utterance for new_resp: {utterance}")
                            break
                        else:
                            print(f"DEBUG new_resp (NRPA Path, Terminal, skipped): Speaker: {speaker}, Utterance: {utterance}")
                            pass
                else:
                    # print(f"NRPA 返回非终止状态，正常提取下一步动作")
                    # 只有在非终止状态下才提取动作
                    current_state_len = len(current_state.history)
                    for i in range(current_state_len, len(final_state.history)):
                        next_turn = final_state.history[i]
                        speaker, da, utterance = next_turn
                        if speaker == EmotionalSupportGame.SYS:
                            nrpa_policy_next_da = da
                            nrpa_pred_rep = utterance
                            print(f"DEBUG new_resp (NRPA Path): Speaker: {speaker}, Utterance for new_resp: {utterance}")
                            break
                        else:
                            print(f"DEBUG new_resp (NRPA Path, skipped): Speaker: {speaker}, Utterance: {utterance}")
                            pass
            else:
                # NRPA未能生成有效结果，使用随机选择
                legal_moves = dialog_planner.legalMoves(current_state)
                if legal_moves:
                    next_da_idx = np.random.choice(legal_moves)
                    nrpa_policy_next_da = system.dialog_acts[next_da_idx]

                    # 检查游戏的get_next_state方法签名以兼容新旧接口
                    import inspect
                    sig = inspect.signature(game.get_next_state)
                    num_params = len(sig.parameters)

                    if num_params <= 2:  # 老接口：get_next_state(state, action)
                        temp_next_state = game.get_next_state(current_state.copy(), next_da_idx)
                    else:  # 新接口：get_next_state(state, action, agent_state, mode)
                        agent_state = []
                        temp_next_state, temp_agent_state, temp_reward = game.get_next_state(current_state.copy(), next_da_idx, agent_state)

                    if len(temp_next_state.history) > len(current_state.history):
                        for i in range(len(current_state.history), len(temp_next_state.history)):
                            turn = temp_next_state.history[i]
                            speaker, da, utterance = turn
                            if speaker == EmotionalSupportGame.SYS:
                                nrpa_pred_rep = utterance
                                break
                        else:
                            nrpa_pred_rep = "Error: Could not find system response"
                    else:
                        nrpa_pred_rep = "Error generating random response"
                else:
                    # print(f"错误：对话 {did} 轮次 {t} 无合法动作可选！")
                    turn_successful = False
                    nrpa_pred_rep = "No legal moves, no response generated."
                    break

            # 获取当前轮次对应的原始系统回应作为human_resp（考虑索引偏移）
            ori_dataset_idx = t + 1  # 跳过被situation替换的前两句，取当前轮次对应的原始系统回应
            if ori_dataset_idx < len(sys_turns):
                human_resp = sys_turns[ori_dataset_idx]["text"].strip()
                next_sys_strategy = sys_turns[ori_dataset_idx].get("strategy", "Others")
                next_sys_da = next_sys_strategy if next_sys_strategy in system.dialog_acts else "Others"
            else:
                human_resp = "Thank you for talking with me."
                next_sys_da = "Others"

            debug_data = {
                "selected_da_idx": next_da_idx,
                "selected_da": nrpa_policy_next_da,
                "final_state_score": dialog_planner.score(final_state,
                                                          len_history_before_nrpa_call) if final_state else None,
                "final_state_history_len": len(final_state.history) if final_state else 0,
                "nrpa_iterations": nrpa_args.nrpa_iterations,
                "nrpa_depth": nrpa_args.nrpa_depth,
                "nrpa_search_time": nrpa_duration,
            }

            cmp_data = {
                'did': did,
                'context': context,
                'ori_resp': human_resp,
                'ori_da': next_sys_da,
                'new_resp': nrpa_pred_rep,
                'new_da': nrpa_policy_next_da,
                'debug': debug_data
            }

            output.append(cmp_data)

        if not turn_successful:
            # print(f"对话 {did} 处理中遇到错误")
            pass

        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)

        processed_dialogs.add(did)
        num_done += 1
        pbar.update(1)
        # print(f"对话 {did} 完成 ({num_done}/{needed_new_dialogs})\n")

    pbar.close()

    # 计算并显示总运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== NRPA 实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))} ===")
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration / 60:.2f}分钟) ===")
    print(f"=== 平均每个对话处理时间: {total_duration / len(processed_dialogs):.2f}秒 ===")

    print(f"\n所有对话处理完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数: {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default=r"C:\Users\Windows11\Desktop\GDPZero-master\outputs\level_测试10035_nrpa_ESC.pkl",
                        help='output file path')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "gpt-3.5-turbo-0613",
                                 "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat",
                                 "qwen2-7b-instruct", "qwen-plus", "qwen-turbo", "qwen-max"],
                        help='LLM backbone model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences for LLM generation')
    parser.add_argument('--nrpa_depth', type=int, default=2, help='NRPA recursive search depth')
    parser.add_argument('--nrpa_iterations', type=int, default=10, help='Number of iterations per NRPA level')
    parser.add_argument('--num_dialogs', type=int, default=10, help='Target number of dialogs to process')
    parser.add_argument('--nrpa_playout_epsilon', type=float, default=0,
                        help='Epsilon for epsilon-greedy exploration in NRPA playouts')
    parser.add_argument('--reduced_iterations', type=int, default=0,
                        help='Reduced number of iterations (overrides nrpa_iterations if > 0)')
    parser.add_argument('--max_playout_steps', type=int, default=0, help='Maximum playout steps (0 means unlimited)')
    parser.add_argument('--start_dialog', type=int, default=1, help='Start processing from dialog number (1-based)')
    parser.add_argument('--early_stopping_enabled', type=bool, default=True, help='Enable early stopping mechanism')
    parser.add_argument('--early_stopping_threshold', type=float, default=1.0035, help='Early stopping threshold')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min_iterations', type=int, default=3, help='Minimum iterations')

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
        print(
            f"  Early Stopping: Enabled, Threshold={cmd_args.early_stopping_threshold}, Patience={cmd_args.early_stopping_patience}, Minimum Iterations={cmd_args.min_iterations}")

    main(cmd_args)
