import numpy as np
import logging
import pickle
import argparse
import os
import time
import json
from tqdm.auto import tqdm
from core.gen_models import (
    LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, 
    GPT35Turbo0613ChatModel, GPT4Turbo20240409ChatModel,
    GPT4oMini20240718ChatModel, DeepSeekChatModel
)
from core.esc_players import (
    TherapistModel, PatientModel, ESCSystemPlanner,
    TherapistChatModel, PatientChatModel, ESCChatSystemPlanner
)
from core.game1 import EmotionalSupportGame
from core.helpers import DialogSession
from utils.prompt_examples import ESConv_EXP_DIALOG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== Raw Prompting ESC 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

    game_ontology = EmotionalSupportGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']
    system_name = EmotionalSupportGame.SYS
    user_name = EmotionalSupportGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(ESConv_EXP_DIALOG)

    if cmd_args.llm == 'code-davinci-002':
        backbone_model = OpenAIModel(cmd_args.llm)
        SysModel = TherapistModel
        UsrModel = PatientModel
        SysPlanner = ESCSystemPlanner
    elif cmd_args.llm in ['gpt-3.5-turbo']:
        backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'gpt-3.5-turbo-0613':
        backbone_model = GPT35Turbo0613ChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'gpt-4-turbo-2024-04-09':
        backbone_model = GPT4Turbo20240409ChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'gpt-4o-mini-2024-07-18':
        backbone_model = GPT4oMini20240718ChatModel(cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'chatgpt':
        backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner
    elif cmd_args.llm == 'deepseek-chat':
        backbone_model = DeepSeekChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = TherapistChatModel
        UsrModel = PatientChatModel
        SysPlanner = ESCChatSystemPlanner

    system = SysModel(
        sys_da,
        backbone_model,
        conv_examples=[exp_1],
        inference_args={
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        },
        zero_shot=False
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
        zero_shot=False
    )
    planner = SysPlanner(
        dialog_acts=system.dialog_acts,
        max_hist_num_turns=system.max_hist_num_turns,
        user_dialog_acts=user.dialog_acts,
        user_max_hist_num_turns=user.max_hist_num_turns,
        generation_model=backbone_model,
        conv_examples=[exp_1],
        zero_shot=False
    )
    game = EmotionalSupportGame(system, user, planner, zero_shot=False)
    
    print(f"使用模型: {cmd_args.llm}")
    print(f"系统对话行为: {system.dialog_acts}")
    print(f"用户对话行为: {user.dialog_acts}")

    # 加载ESC数据集 - 与gdpzero_ESC.py相同
    all_dialogs = {}
    with open(r"/Users/joey/Desktop/GDPZero-master/data/esc-train.txt", "r", encoding="utf-8") as f:
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

    output = []
    processed_dialogs = set()
    
    # 检查是否存在现有输出文件
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
    else:
        print(f"输出文件 {cmd_args.output} 不存在，将创建新文件")

    bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']
    dialog_keys_to_process = [k for k in all_dialogs.keys() if k not in bad_dialogs]
    target_dialogs_count = min(cmd_args.num_dialogs, len(dialog_keys_to_process))

    needed_new_dialogs = target_dialogs_count - len(processed_dialogs)

    if needed_new_dialogs <= 0:
        print(f"已完成所有 {target_dialogs_count} 个目标对话的处理")
        return

    print(
        f"目标处理 {target_dialogs_count} 个对话，当前已处理 {len(processed_dialogs)} 个，将处理 {needed_new_dialogs} 个新对话")
    
    num_done = 0
    pbar = tqdm(total=needed_new_dialogs, desc="evaluating")

    for did in dialog_keys_to_process:
        if did in processed_dialogs:
            continue

        if num_done >= needed_new_dialogs:
            print(f"已完成 {needed_new_dialogs} 个新对话的处理目标")
            break

        print(f"\n评估对话ID: {did} (新对话 {num_done + 1}/{needed_new_dialogs})")
        dialog = all_dialogs[did]
        
        # 从对话数据中提取情感类型和问题类型，如果没有则使用默认值
        emotion_type = dialog.get("emotion_type", "unknown")
        problem_type = dialog.get("problem_type", "unknown")
        
        print(f"情感类型: {emotion_type}, 问题类型: {problem_type}")
        
        # 获取situation信息
        situation = dialog.get("situation", "")
        print(f"情况描述: {situation}")
        
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
            print(f"对话 {did} 没有足够的轮次")
            continue
            
        # 逐轮处理对话，与gdpzero_ESC.py保持相同的逻辑
        for t in range(max_pairs - 1):  # Skip last turn
            print(f"\n=== 第 {t+1} 轮对话开始 ===")
            
            # 构建当前轮次的状态（基于原始数据集）
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
                            print(f"  警告: 使用PatientChatModel生成用户DA失败: {e}")
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
                    
                    # 防止在初始对话状态中出现Solved，将其替换为Feel better
                    if usr_da_i == EmotionalSupportGame.U_Solved:
                        print(f"  警告: 第 {i+1} 轮用户DA为Solved，替换为Feel better")
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
                print(f"对话 {did} 在第 {t} 轮结束，没有更多用户回应")
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

            # 使用raw prompting获取系统预测（而不是MCTS）
            try:
                prompt_start_time = time.time()
                
                # 使用planner获取prior和value
                prior, v = planner.predict(current_state)
                greedy_policy = system.dialog_acts[np.argmax(prior)]
                
                # 获取下一个状态以生成回应
                next_best_state = game.get_next_state(current_state, np.argmax(prior))
                greedy_pred_resp = next_best_state.history[-2][2]
                
                prompt_end_time = time.time()
                prompt_duration = prompt_end_time - prompt_start_time
                print(f"Raw prompting 耗时: {prompt_duration:.2f}秒")
                
            except Exception as e:
                print(f"Raw prompting 生成失败: {e}")
                bad_dialogs.append(did)
                continue

            # 获取当前轮次对应的原始系统回应作为human_resp（考虑索引偏移）
            ori_dataset_idx = t + 1  # 跳过被situation替换的前两句，取当前轮次对应的原始系统回应
            if ori_dataset_idx < len(sys_turns):
                human_resp = sys_turns[ori_dataset_idx]["text"].strip()
                next_sys_strategy = sys_turns[ori_dataset_idx].get("strategy", "Others")
                next_sys_da = next_sys_strategy if next_sys_strategy in system.dialog_acts else "Others"
            else:
                human_resp = "Thank you for talking with me."
                next_sys_da = "Others"

            # logging for debug
            debug_data = {
                "prior": prior,
                "da": greedy_policy,
                "v": v,
                "prompt_time": prompt_duration,
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
    print(f"\n=== Raw Prompting ESC 实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))} ===")
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration / 60:.2f}分钟) ===")
    print(f"=== 平均每个对话处理时间: {total_duration / len(processed_dialogs):.2f}秒 ===")

    print(f"\n所有对话处理完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数: {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "gpt-3.5-turbo-0613",
                                 "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat"],
                        help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1,
                        help='max number of sentences to generate. -1 for no limit')
    parser.add_argument('--output', type=str,
                        default=r"/Users/joey/Desktop/GDPZero-master/outputs/gpt-4o-mini-2024-07-18_raw_prompt_ESC_10dialogs.pkl",
                        help='output file')
    parser.add_argument('--num_dialogs', type=int, default=10, help='Target number of dialogs to process')
    cmd_args = parser.parse_args()
    print("saving to", cmd_args.output)
    print(f"LLM Model: {cmd_args.llm}")
    print(f"Max Gen Sentences: {cmd_args.gen_sentences}")
    print(f"Target Dialogs: {cmd_args.num_dialogs}")

    main(cmd_args)