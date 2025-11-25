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
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== GDPZero 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")
    
    game_ontology = PersuasionGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']
    system_name = PersuasionGame.SYS
    user_name = PersuasionGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

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
    print(f"使用模型: {cmd_args.llm}")
    print(f"系统对话行为: {system.dialog_acts}")
    print(f"用户对话行为: {user.dialog_acts}")

    with open("C:\\Users\\Windows11\\Desktop\\GDPZero-master\\data\\p4g\\300_dialog_turn_based.pkl", "rb") as f:
        all_dialogs = pickle.load(f)

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

    print(f"目标处理 {target_dialogs_count} 个对话，当前已处理 {len(processed_dialogs)} 个，将处理 {needed_new_dialogs} 个新对话")
    pbar = tqdm(total=needed_new_dialogs, desc="evaluating")
    
    num_done = 0

    for did in dialog_keys_to_process:
        if did in processed_dialogs:
            print(f"跳过已处理的对话ID: {did}")
            continue
        
        if num_done >= needed_new_dialogs:
            print(f"已完成 {needed_new_dialogs} 个新对话的处理目标")
            break

        print(f"正在评估对话ID: {did} (新对话 {num_done+1}/{needed_new_dialogs})")
        context = ""
        dialog = all_dialogs[did]
        state = game.init_dialog()
        
        print(f"初始状态历史长度: {len(state.history)}")

        for t, turn in enumerate(dialog["dialog"]):
            if len(turn["ee"]) == 0:  # Ended
                print(f"对话 {did} 在第 {t} 轮结束，用户没有回应")
                break
            if t == len(dialog["dialog"]) - 1:  # Skip last turn
                print(f"对话 {did} 跳过最后一轮")
                break

            usr_utt = " ".join(turn["ee"]).strip()
            usr_da = dialog["label"][t]["ee"][-1]

            # Map user dialog act
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

            if usr_da == PersuasionGame.U_Donate:
                print(f"对话 {did} 在第 {t} 轮结束，用户同意捐款")
                break

            sys_utt = " ".join(turn["er"]).strip()
            sys_da = set(dialog["label"][t]["er"])
            intersected_das = sys_da.intersection(system.dialog_acts)
            if len(intersected_das) == 0:
                sys_da = "other"
            else:
                sys_da = list(intersected_das)[-1]

            state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
            state.add_single(PersuasionGame.USR, usr_da, usr_utt)
            
            context = f"""
            {context}
            Persuader: {sys_utt}
            Persuadee: {usr_utt}
            """
            context = context.replace('\t', '').strip()

            if isinstance(backbone_model, OpenAIModel):
                backbone_model._cached_generate.cache_clear()
            elif hasattr(backbone_model, '_cached_generate'):
                backbone_model._cached_generate.cache_clear()
            
            # 记录MCTS搜索时间
            mcts_start_time = time.time()
            dialog_planner = OpenLoopMCTS(game, planner, args)

            for i in tqdm(range(args.num_MCTS_sims)):
                v = dialog_planner.search(state)
            
            mcts_end_time = time.time()
            mcts_duration = mcts_end_time - mcts_start_time
            print(f"MCTS搜索耗时: {mcts_duration:.2f}秒 ({args.num_MCTS_sims}次模拟)")

            mcts_policy = dialog_planner.get_action_prob(state)
            best_action_idx = np.argmax(mcts_policy)
            mcts_policy_next_da = system.dialog_acts[best_action_idx]
            mcts_pred_rep = dialog_planner.get_best_realization(state, best_action_idx)

            human_resp = " ".join(dialog["dialog"][t+1]["er"]).strip()
            next_sys_das = set(dialog["label"][t+1]["er"])
            next_intersected_das = next_sys_das.intersection(system.dialog_acts)
            if len(next_intersected_das) == 0:
                next_sys_da = "other"
            else:
                next_sys_da = list(next_intersected_das)[-1]

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
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟) ===")
    print(f"=== 平均每个对话处理时间: {total_duration/len(processed_dialogs):.2f}秒 ===")
    
    print(f"所有目标对话评估完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数 {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="C:\\Users\\Windows11\\Desktop\\GDPZero-master\\outputs\\dpzero_50sim_20.pkl", help='output file')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat"], help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=3, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=1, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for uninitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=1, help='number of dialogs to test MCTS on')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    cmd_args = parser.parse_args()
    print("保存结果到", cmd_args.output)

    main(cmd_args)
