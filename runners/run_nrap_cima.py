import numpy as np
import logging
import pickle
import argparse
import os
import time

from tqdm.auto import tqdm

from core.gen_models import (
    LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, GPT35Turbo0613ChatModel, GPT4Turbo20240409ChatModel,
    GPT4oMini20240718ChatModel, DeepSeekChatModel, GPT4oMinChatModel, DashScopeChatModel,
    Qwen2_7B_InstructChatModel, QwenPlusChatModel, QwenTurboChatModel, QwenMaxChatModel,
    LocalOpenAIChatModel, LocalQwenChatModel, LocalLlamaChatModel, LocalChatGLMChatModel
)
from core.players import (
    PersuadeeModel, PersuaderModel, P4GSystemPlanner,
    PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.esc_players import (
    TherapistModel, PatientModel, ESCSystemPlanner,
    TherapistChatModel, PatientChatModel, ESCChatSystemPlanner
)

from core.cima_palyers import (
    TeacherChatModel, StudentChatModel, CIMAChatSystemPlanner

)
from core.cb_players import (
    BuyerModel, SellerModel, CBSystemPlanner,
    BuyerChatModel, SellerChatModel, CBChatSystemPlanner
)
from core.game1 import EmotionalSupportGame, CBGame,CIMAGame
from core.game import PersuasionGame
from core.helpers import DialogSession, CBDialogSession, CIMADialogSession
from utils.utils import dotdict
from utils.prompt_examples import ESConv_EXP_DIALOG, CB_EXP_DIALOG, CIMA_EXP_DIALOG

from core.nrpa_cima import NRPAPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 为本地模型设置日志级别
logging.getLogger('core.gen_models').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== NRPA 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

    system_name = CIMAGame.SYS
    user_name = CIMAGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(CIMA_EXP_DIALOG)

    game_ontology = CIMAGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']

    if cmd_args.llm == 'code-davinci-002':
        backbone_model = OpenAIModel(cmd_args.llm)
        SysModel = BuyerModel
        UsrModel = SellerModel
        SysPlanner = CBSystemPlanner
    elif cmd_args.llm in ['gpt-3.5-turbo']:
        backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
    elif cmd_args.llm == 'chatgpt':
        backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'gpt-3.5-turbo-0613':
        backbone_model = GPT35Turbo0613ChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'gpt-4-turbo-2024-04-09':
        backbone_model = GPT4Turbo20240409ChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'gpt-4o-mini-2024-07-18':
        backbone_model = GPT4oMini20240718ChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'gpt-4o-mini':
        backbone_model = GPT4oMinChatModel(cmd_args.gen_sentences)
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
    elif cmd_args.llm == 'deepseek-chat':
        backbone_model = DeepSeekChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'qwen2-7b-instruct':
        backbone_model = Qwen2_7B_InstructChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'qwen2.5-7b-instruct':
        backbone_model = QwenPlusChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'qwen3-8b':
        backbone_model = QwenTurboChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'qwen3-0.6b':
        backbone_model = QwenMaxChatModel(cmd_args.gen_sentences)
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
    elif cmd_args.llm == 'local-openai':
        # 通用本地 OpenAI 兼容模型
        backbone_model = LocalOpenAIChatModel(
            model_name=getattr(cmd_args, 'local_model_name', 'xxx'),
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1'),
            gen_sentences=cmd_args.gen_sentences
        )
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
    elif cmd_args.llm == 'local-qwen':
        # 本地 Qwen 模型
        backbone_model = LocalQwenChatModel(
            gen_sentences=cmd_args.gen_sentences,
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1')
        )
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
    elif cmd_args.llm == 'local-llama':
        # 本地 Llama 模型
        backbone_model = LocalLlamaChatModel(
            gen_sentences=cmd_args.gen_sentences,
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1')
        )
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
    elif cmd_args.llm == 'local-chatglm':
        # 本地 ChatGLM 模型
        backbone_model = LocalChatGLMChatModel(
            gen_sentences=cmd_args.gen_sentences,
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1')
        )
        SysModel = TeacherChatModel
        UsrModel = StudentChatModel
        SysPlanner = CIMAChatSystemPlanner
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

    game = CIMAGame(system, user, planner, zero_shot=False)
    print(f"使用模型: {cmd_args.llm}")
    print(f"系统对话行为: {system.dialog_acts}")
    print(f"用户对话行为: {user.dialog_acts}")

    import json
    all_dialogs = {}
    with open(r"D:\GDPZero-master\data\cima-test.txt", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    dialog_data = json.loads(line)
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
        "early_stopping_enabled": cmd_args.early_stopping_enabled,
        "early_stopping_threshold": cmd_args.early_stopping_threshold,
        "early_stopping_patience": cmd_args.early_stopping_patience,
        "min_iterations": cmd_args.min_iterations,
        "debug": cmd_args.debug,
    })
    
    print(f"调试输出: 已启用")

    output = []
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
            print(f"读取输出文件失败: {e}. 将创建新的输出文件.")
    else:
        print(f"输出文件 {cmd_args.output} 不存在，将创建新文件")

    bad_dialogs = [
        # Add any problematic CIMA dialog IDs here if needed
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

    total_turns = 0
    successful_dialogs = 0
    dialog_turn_counts = []
    successful_turn_counts = []

    dialog_count = 0
    for did in dialog_keys_to_process:
        if did in processed_dialogs:
            continue
        if num_done >= needed_new_dialogs:
            break

        dialog_count += 1
        if hasattr(cmd_args, 'start_dialog') and dialog_count < cmd_args.start_dialog:
            print(f"跳过对话 {dialog_count}: {did}")
            continue

        print(f"\n正在评估对话ID: {did} ({num_done + 1}/{needed_new_dialogs})")
        dialog = all_dialogs[did]

        # CIMA对话场景的数据结构
        sentence = dialog.get("sentence", "")
        target = dialog.get("target", "")
        existing_dialog = dialog.get("dialog", [])

        logger.info("evaluating dialog sentence: {}".format(sentence))
        
        # 将现有对话转换为历史格式 - 只取前两句，按照env.py的逻辑
        history = []
        if existing_dialog and len(existing_dialog) >= 2:
            # 只取前两句对话
            first_turn = existing_dialog[0]  # 系统的第一句
            second_turn = existing_dialog[1]  # 用户的第一句回应
            
            if first_turn.get("speaker") == "sys":
                history.append((CIMAGame.SYS, CIMAGame.S_Others, first_turn.get("text", "")))
            
            if second_turn.get("speaker") == "usr":
                history.append((CIMAGame.USR, CIMAGame.U_DidNotTry, second_turn.get("text", "")))
        
        initial_state = game.init_dialog(sentence, target, history)

        # 按照CIMA的初始对话场景
        sys_role = CIMAGame.SYS
        usr_role = CIMAGame.USR
        
        # 如果没有现有的对话历史，创建初始对话
        if not history:
            initial_history = [(sys_role, CIMAGame.S_Others, f"Please translate \"{sentence}\" into Italian.")]
            initial_state = game.init_dialog(sentence, target, initial_history)
        
        # 从历史记录中获取上下文用于打印
        if initial_state.history:
            sys_utt = initial_state.history[0][2]
            if len(initial_state.history) > 1:
                usr_utt = initial_state.history[1][2]
            else:
                usr_utt = ""
        else:
            sys_utt = f"Please translate \"{sentence}\" into Italian."
            usr_utt = ""
        
        end_condition = CIMAGame.U_Correct

        context = f"""
        {sys_role}: {sys_utt}
        {usr_role}: {usr_utt}
        """
        initial_context = context.replace('\t', '').strip()
        print(f"\n=== 开始模拟对话 {did} ===")
        print(f"初始对话上下文:\n{initial_context}\n" + "=" * 50)
        print(f"要翻译的句子: {sentence}")
        print(f"目标翻译: {target}")

        # 清理缓存
        if hasattr(backbone_model, '_cached_generate'):
            backbone_model._cached_generate.cache_clear()
        if hasattr(system, '_cached_generate'):
            system._cached_generate.cache_clear()
        if hasattr(user, '_cached_generate'):
            user._cached_generate.cache_clear()
        if hasattr(planner, '_cached_generate'):
            planner._cached_generate.cache_clear()

        # --- 一次性完整对话模拟 ---
        print(f"开始NRPA搜索 (深度={nrpa_args.nrpa_depth}, 迭代={nrpa_args.nrpa_iterations})...")
        nrpa_start_time = time.time()
        dialog_planner = NRPAPlanner(game, planner, nrpa_args)
        final_state = dialog_planner.nrpa(nrpa_args.nrpa_depth, {}, initial_state.copy())
        nrpa_duration = time.time() - nrpa_start_time
        print(f"NRPA搜索完成! 耗时: {nrpa_duration:.2f}秒")

        # --- 处理模拟结果 ---
        if final_state and len(final_state.history) > len(initial_state.history):
            print("\n--- 对话模拟详细过程 ---")
            simulated_turns = final_state.history[len(initial_state.history):]
            current_context = sys_utt + "\n" + usr_utt
            is_solved = False

            turn_count_in_sim = 0
            for i in range(0, len(simulated_turns), 2):
                turn_count_in_sim += 1

                sys_turn = simulated_turns[i]
                sys_da, sys_resp = sys_turn[1], sys_turn[2]

                if i + 1 < len(simulated_turns):
                    usr_turn = simulated_turns[i + 1]
                    usr_da, usr_resp = usr_turn[1], usr_turn[2]
                else:
                    # 对话以系统回应结束
                    usr_da, usr_resp = "N/A", ""

                print(f"\n--- 模拟轮次: {turn_count_in_sim} ---")
                print(f"Teacher: [{sys_da}] {sys_resp}")
                print(f"Student: [{usr_da}] {usr_resp}")

                current_context += f"\nTeacher: {sys_resp}\nStudent: {usr_resp}"

                cmp_data = {
                    'did': did,
                    'turn': turn_count_in_sim,
                    'context': current_context.strip(),
                    'new_resp': sys_resp,
                    'new_da': sys_da,
                    'usr_resp': usr_resp,
                    'usr_da': usr_da,
                    'sentence': sentence,
                    'target': target,
                    "debug": {
                        "nrpa_iterations": nrpa_args.nrpa_iterations,
                        "nrpa_depth": nrpa_args.nrpa_depth,
                        "nrpa_search_time": nrpa_duration,
                    }
                }
                output.append(cmp_data)

                if usr_da == CIMAGame.U_Correct:
                    is_solved = True
                    break

            print("-" * 50)

            if is_solved:
                print(f"\n🎉 对话 {did} 在第 {turn_count_in_sim} 轮结束 (学生正确翻译)!")
                successful_dialogs += 1
                successful_turn_counts.append(turn_count_in_sim)
            else:
                print(f"\n❌ 对话 {did} 模拟结束时学生未能正确翻译 (共 {turn_count_in_sim} 轮)")

            dialog_turn_counts.append(turn_count_in_sim)
            total_turns += turn_count_in_sim

        else:
            print("警告: NRPA未能生成有效对话。")
            dialog_turn_counts.append(0)

        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)

        processed_dialogs.add(did)
        num_done += 1
        pbar.update(1)

    pbar.close()

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== NRPA 实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))} ===")
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration / 60:.2f}分钟) ===")
    if num_done > 0:
        print(f"=== 平均每个对话处理时间: {total_duration / num_done:.2f}秒 ===")

    if num_done > 0:
        success_rate = (successful_dialogs / num_done) * 100
        average_turns = total_turns / num_done
        print(f"\n=== 对话统计结果 ===")
        print(f"总对话数: {num_done}")
        print(f"成功学会翻译的对话数: {successful_dialogs}")
        print(f"成功率: {success_rate:.1f}% ({successful_dialogs}/{num_done})")
        print(f"平均轮数: {average_turns:.1f}轮")
        print(f"各对话轮数分布: {dialog_turn_counts}")
        if successful_dialogs > 0:
            avg_successful_turns = sum(successful_turn_counts) / len(successful_turn_counts)
            print(f"成功学习的平均轮数: {avg_successful_turns:.1f}轮")
            print(f"成功学习的轮数分布: {successful_turn_counts}")

    print(f"\n所有对话处理完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数: {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default=r"D:\GDPZero-master\outputs\gpt-3.5-turbo_nrpa_level2_CIMA_test.pkl",
                        help='output file path')
    parser.add_argument('--llm', type=str, default="gpt-3.5-turbo",
                        choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "gpt-3.5-turbo-0613",
                                 "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat", "gpt-4o-mini",
                                 "qwen2-7b-instruct", "qwen2.5-7b-instruct", "qwen3-8b", "qwen3-0.6b",
                                 "local-openai", "local-qwen", "local-llama", "local-chatglm"],
                        help='LLM backbone model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences for LLM generation')
    parser.add_argument('--nrpa_depth', type=int, default=2, help='NRPA recursive search depth. 0 for pure playout.')
    parser.add_argument('--nrpa_iterations', type=int, default=3, help='Number of iterations per NRPA level')
    parser.add_argument('--num_dialogs', type=int, default=130, help='Target number of dialogs to process')
    parser.add_argument('--nrpa_playout_epsilon', type=float, default=0,
                        help='Epsilon for epsilon-greedy exploration in NRPA playouts')
    parser.add_argument('--reduced_iterations', type=int, default=0,
                        help='Reduced number of iterations (overrides nrpa_iterations if > 0)')
    parser.add_argument('--max_playout_steps', type=int, default=10, help='Maximum playout steps (0 means unlimited)')
    parser.add_argument('--start_dialog', type=int, default=1, help='Start processing from dialog number (1-based)')
    parser.add_argument('--early_stopping_enabled', type=bool, default=True, help='Enable early stopping mechanism')
    parser.add_argument('--early_stopping_threshold', type=int, default=2,
                        help='Early stopping threshold (dialog turns)')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--min_iterations', type=int, default=1, help='Minimum iterations')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
    # 本地模型相关参数
    parser.add_argument('--local_base_url', type=str, default='http://localhost:6006/v1', 
                        help='Base URL for local OpenAI-compatible API')
    parser.add_argument('--local_model_name', type=str, default='xxx', 
                        help='Model name for local OpenAI-compatible API')

    cmd_args = parser.parse_args()

    # 安全检查：如果深度为0，强制进行playout
    if cmd_args.nrpa_depth == 0:
        print("警告: NRPA 深度为0，将仅使用 playout 进行模拟，无递归搜索。")

    print("\n命令行参数:")
    print(f"  Output file: {cmd_args.output}")
    print(f"  LLM Model: {cmd_args.llm}")
    print(f"  Max Gen Sentences: {cmd_args.gen_sentences}")
    print(f"  NRPA Depth: {cmd_args.nrpa_depth}")
    print(f"  NRPA Iterations: {cmd_args.nrpa_iterations}")
    print(f"  Num Dialogs: {cmd_args.num_dialogs}")
    print(f"  NRPA Playout Epsilon: {cmd_args.nrpa_playout_epsilon}")
    print(f"  Max Playout Steps: {cmd_args.max_playout_steps}")
    print(f"  Start Dialog: {cmd_args.start_dialog}")
    if cmd_args.reduced_iterations > 0:
        print(f"  Using Reduced Iterations: {cmd_args.reduced_iterations}")
    if cmd_args.early_stopping_enabled:
        print(
            f"  Early Stopping: Enabled, Threshold={cmd_args.early_stopping_threshold}, Patience={cmd_args.early_stopping_patience}, Minimum Iterations={cmd_args.min_iterations}")
    
    # 显示本地模型配置
    if cmd_args.llm.startswith('local-'):
        print(f"  Local Model Base URL: {cmd_args.local_base_url}")
        print(f"  Local Model Name: {cmd_args.local_model_name}")

    main(cmd_args)
