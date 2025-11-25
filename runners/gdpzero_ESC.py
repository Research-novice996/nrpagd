import numpy as np
import logging
import pickle
import argparse
import os
import time
from tqdm.auto import tqdm
from core.gen_models import (
    LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel,
    GPT35Turbo0613ChatModel, GPT4Turbo20240409ChatModel, GPT4oMini20240718ChatModel, DeepSeekChatModel
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
from core.game1 import EmotionalSupportGame, CBGame
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG
from utils.prompt_examples import ESConv_EXP_DIALOG, CB_EXP_DIALOG
# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== GDPZero 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

    game_ontology = EmotionalSupportGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']
    system_name = EmotionalSupportGame.SYS
    user_name = EmotionalSupportGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(ESConv_EXP_DIALOG)

    if cmd_args.llm in ['code-davinci-002']:
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
        SysModel = TherapistChatModel
        UsrModel =  PatientChatModel
        SysPlanner = ESCChatSystemPlanner
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

    system = SysModel(
        sys_da,
        backbone_model,
        conv_examples=[exp_1],
        inference_args={
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        },
        zero_shot=False # 确保与游戏设置一致
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
        zero_shot= False# 确保与游戏设置一致
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
    args = dotdict({
        "cpuct": 1.0,
        "num_MCTS_sims": cmd_args.num_mcts_sims,
        "Q_0": cmd_args.Q_0,
        "max_realizations": cmd_args.max_realizations,
    })

    output = []
    bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']
    num_done = 0
    pbar = tqdm(total=num_dialogs, desc="evaluating")

    print(f"准备评估 {num_dialogs} 个对话")

    processed_dialogs = set()
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
            output = []
            processed_dialogs = set()
    else:
        print(f"输出文件 {cmd_args.output} 不存在，将创建新文件")
        output = []
        processed_dialogs = set()

    dialog_keys_to_process = [k for k in all_dialogs.keys() if k not in bad_dialogs]
    target_dialogs_count = min(num_dialogs, len(dialog_keys_to_process))

    needed_new_dialogs = target_dialogs_count - len(processed_dialogs)

    if needed_new_dialogs <= 0:
        print(f"已完成所有 {target_dialogs_count} 个目标对话的处理")
        pbar.close()
        return

    print(
        f"目标处理 {target_dialogs_count} 个对话，当前已处理 {len(processed_dialogs)} 个，将处理 {needed_new_dialogs} 个新对话")
    pbar = tqdm(total=needed_new_dialogs, desc="evaluating")

    num_done = 0

    for did in dialog_keys_to_process:
        if did in processed_dialogs:
            print(f"跳过已处理的对话ID: {did}")
            continue

        if num_done >= needed_new_dialogs:
            print(f"已完成 {needed_new_dialogs} 个新对话的处理目标")
            break

        print(f"正在评估对话ID: {did} (新对话 {num_done + 1}/{needed_new_dialogs})")
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
            
        # 逐轮处理对话，使用PatientChatModel生成正确的对话行为
        for t in range(max_pairs - 1):  # Skip last turn
            print(f"\n=== 第 {t+1} 轮对话开始 ===")
            
            # 构建当前轮次的状态（基于原始数据集，不受MCTS影响）
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
            
            # # 打印当前轮次的初始输入
            # print(f"第 {t+1} 轮初始输入:")
            # print(f"  用户话语: {usr_utt}")
            # print(f"  原始系统话语: {sys_utt}")
            # print(f"  原始系统策略: {sys_da_final}")

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

            if isinstance(backbone_model, OpenAIModel):
                backbone_model._cached_generate.cache_clear()
            elif hasattr(backbone_model, '_cached_generate'):
                backbone_model._cached_generate.cache_clear()

            # 记录MCTS搜索时间
            mcts_start_time = time.time()
            dialog_planner = OpenLoopMCTS(game, planner, args)
            
            for i in tqdm(range(args.num_MCTS_sims)):
                v = dialog_planner.search(current_state)

            mcts_end_time = time.time()
            mcts_duration = mcts_end_time - mcts_start_time
            print(f"MCTS搜索耗时: {mcts_duration:.2f}秒 ({args.num_MCTS_sims}次模拟)")

            mcts_policy = dialog_planner.get_action_prob(current_state)
            best_action_idx = np.argmax(mcts_policy)
            mcts_policy_next_da = system.dialog_acts[best_action_idx]
            mcts_pred_rep = dialog_planner.get_best_realization(current_state, best_action_idx)

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
                "probs": mcts_policy,
                "da": mcts_policy_next_da,
                "mcts_search_time": mcts_duration,
                "num_mcts_sims": args.num_MCTS_sims,
                "search_tree": {
                    "Ns": dialog_planner.Ns,
                    "Nsa": dialog_planner.Nsa,
                    "Q": dialog_planner.Q,
                    "P": dialog_planner.P,
                    "Vs": dialog_planner.Vs,
                    "realizations": dialog_planner.realizations,
                    "realizations_Vs": dialog_planner.realizations_Vs,
                    "realizations_Ns": dialog_planner.realizations_Ns,
                },
            }

            cmp_data = {
                'did': did,
                'context': context,
                'ori_resp': human_resp,
                'ori_da': next_sys_da,
                'new_resp': mcts_pred_rep,
                'new_da': mcts_policy_next_da,
                "debug": debug_data,
            }
            output.append(cmp_data)

        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)

        processed_dialogs.add(did)
        num_done += 1
        pbar.update(1)

    pbar.close()

    # 计算并显示总运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== GDPZero 实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))} ===")
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration / 60:.2f}分钟) ===")
    print(f"=== 平均每个对话处理时间: {total_duration / len(processed_dialogs):.2f}秒 ===")

    print(f"所有目标对话评估完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数 {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default=r"C:\Users\Windows11\Desktop\GDPZero-master\outputs\3_50_MCTS_esc.pkl",
                        help='output file')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo", "gpt-3.5-turbo-0613",
                                 "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat"],
                        help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1,
                        help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=50, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0,
                        help='initial Q value for uninitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=10, help='number of dialogs to test MCTS on')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    cmd_args = parser.parse_args()
    print("保存结果到", cmd_args.output)

    main(cmd_args)
