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
    print(f"=== GDPZero P4G模拟实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

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

    with open("/Users/joey/Desktop/GDPZero-master/data/p4g/300_dialog_turn_based.pkl", "rb") as f:
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
    
    max_step = 10  # 设置最大对话轮数
    
    # 统计变量
    total_turns = 0
    successful_dialogs = 0
    dialog_turn_counts = []
    successful_turn_counts = []

    print(f"准备模拟 {num_dialogs} 个对话")

    dialog_keys_to_process = [k for k in all_dialogs.keys() if k not in bad_dialogs]

    for did in dialog_keys_to_process:
        if num_done >= num_dialogs:
            break

        print(f"正在模拟对话ID: {did} ({num_done + 1}/{num_dialogs})")
        dialog = all_dialogs[did]
        
        # 检查对话是否有足够的轮次
        if len(dialog["dialog"]) < 1:
            print(f"跳过对话 {did}：对话轮次不足")
            continue
            
        # 初始化游戏状态
        state = game.init_dialog()

        # 读取前两句话作为初始状态
        first_turn = dialog["dialog"][0]
        
        # 获取第一轮的系统回应和用户回应
        if len(first_turn["er"]) == 0 or len(first_turn["ee"]) == 0:
            print(f"跳过对话 {did}：第一轮对话内容为空")
            continue
            
        sys_utt = " ".join(first_turn["er"]).strip()
        usr_utt = " ".join(first_turn["ee"]).strip()
        
        # 获取对话行为
        sys_da = set(dialog["label"][0]["er"])
        intersected_das = sys_da.intersection(system.dialog_acts)
        if len(intersected_das) == 0:
            sys_da = "other"
        else:
            sys_da = list(intersected_das)[-1]
            
        usr_da = dialog["label"][0]["ee"][-1]
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

        # 添加初始状态
        state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
        state.add_single(PersuasionGame.USR, usr_da, usr_utt)

        context = f"""
        Persuader: {sys_utt}
        Persuadee: {usr_utt}
        """
        context = context.replace('\t', '').strip()

        print(f"\n=== 开始模拟对话 {did} ===")
        print(f"初始对话上下文:")
        print(context)
        print("="*50)
        
        # 如果用户已经同意捐款，直接结束
        if usr_da == PersuasionGame.U_Donate:
            print(f"🎉 对话 {did} 用户已经同意捐款!")
            successful_dialogs += 1
            dialog_turn_counts.append(1)
            successful_turn_counts.append(1)
            total_turns += 1
            
            cmp_data = {
                'did': did,
                'turn': 0,
                'context': context,
                'new_resp': sys_utt,
                'new_da': sys_da,
                'usr_resp': usr_utt,
                'usr_da': usr_da,
                "debug": {"initial_success": True},
            }
            output.append(cmp_data)
            
            with open(cmd_args.output, "wb") as f:
                pickle.dump(output, f)
            num_done += 1
            pbar.update(1)
            continue
        
        # 开始模拟对话
        for t in range(max_step):
            print(f"\n--- 第 {t+1} 轮对话开始 ---")
            
            # 显示当前轮次开始前的对话状态
            print(f"模拟前的对话状态:")
            for i, (role, da, utt) in enumerate(state.history):
                role_name = "Persuader" if role == PersuasionGame.SYS else "Persuadee"
                print(f"  [{i+1}] {role_name}: [{da}] {utt}")
            print()
            
            # 清除缓存
            if isinstance(backbone_model, OpenAIModel):
                backbone_model._cached_generate.cache_clear()
            elif hasattr(backbone_model, '_cached_generate'):
                backbone_model._cached_generate.cache_clear()
            if hasattr(system, '_cached_generate'):
                system._cached_generate.cache_clear()
            if hasattr(user, '_cached_generate'):
                user._cached_generate.cache_clear()
            if hasattr(planner, '_cached_generate'):
                planner._cached_generate.cache_clear()

            print(f"开始MCTS搜索 ({args.num_MCTS_sims} 次模拟)...")
            dialog_planner = OpenLoopMCTS(game, planner, args)
            
            # 运行MCTS搜索
            mcts_start_time = time.time()
            for i in range(args.num_MCTS_sims):
                print(f"第 {i+1} 次模拟...")
                v = dialog_planner.search(state)
            mcts_end_time = time.time()
            mcts_duration = mcts_end_time - mcts_start_time
            print(f"MCTS搜索完成! 耗时: {mcts_duration:.2f}秒")
            
            # 获取MCTS策略和最佳动作
            mcts_policy = dialog_planner.get_action_prob(state)
            best_action_idx = np.argmax(mcts_policy)
            mcts_policy_next_da = system.dialog_acts[best_action_idx]
            
            print(f"系统策略分布: {dict(zip(system.dialog_acts, mcts_policy))}")
            print(f"选择的策略: {mcts_policy_next_da} (概率: {mcts_policy[best_action_idx]:.3f})")
            
            # 获取下一个状态
            next_state = dialog_planner._get_next_state(state, best_action_idx)
            
            # 检查新状态的变化，获取新增的系统和用户回应
            if len(next_state.history) >= len(state.history) + 2:
                # 新增的应该是系统回应和用户回应
                new_sys_turn = next_state.history[-2]  # (role, da, utt)
                new_usr_turn = next_state.history[-1]  # (role, da, utt)
                
                if new_sys_turn[0] == PersuasionGame.SYS and new_usr_turn[0] == PersuasionGame.USR:
                    mcts_pred_rep = new_sys_turn[2]  # 系统回应内容
                    usr_da_next = new_usr_turn[1]    # 用户对话行为
                    usr_utt_next = new_usr_turn[2]   # 用户回应内容
                    
                    print(f"生成的系统回应: {mcts_pred_rep}")
                    print(f"生成的用户回应: {usr_utt_next}")
                    print(f"用户对话行为: {usr_da_next}")
                    
                    # 更新state为新的状态
                    state = next_state
                else:
                    print("警告: 状态角色顺序异常")
                    mcts_pred_rep = "System response error"
                    usr_da_next = PersuasionGame.U_Neutral
                    usr_utt_next = "User response error"
            else:
                print("警告: 状态更新数量异常")
                print(f"原状态长度: {len(state.history)}, 新状态长度: {len(next_state.history)}")
                mcts_pred_rep = "System response error"
                usr_da_next = PersuasionGame.U_Neutral
                usr_utt_next = "User response error"

            context = f"""
            {context}
            Persuader: {mcts_pred_rep}
            Persuadee: {usr_utt_next}
            """
            context = context.replace('\t', '').strip()

            print(f"\n模拟后的完整对话:")
            print(context)
            print("-"*50)

            # logging for debug
            debug_data = {
                "probs": mcts_policy,
                "da": mcts_policy_next_da,
                "mcts_search_time": mcts_duration,
                "num_mcts_sims": args.num_MCTS_sims,
            }

            # update data
            cmp_data = {
                'did': did,
                'turn': t + 1,  # 从第1轮开始计数
                'context': context,
                'new_resp': mcts_pred_rep,
                'new_da': mcts_policy_next_da,
                'usr_resp': usr_utt_next,
                'usr_da': usr_da_next,
                "debug": debug_data,
            }
            output.append(cmp_data)

            if cmd_args.debug:
                logger.info(context)
                logger.info("mcts resp: {}".format(mcts_pred_rep))
                logger.info("mcts da: {}".format(mcts_policy_next_da))

            # 检查是否达到成功条件
            if usr_da_next == PersuasionGame.U_Donate:
                print(f"\n🎉 对话 {did} 在第 {t+1} 轮结束 (用户同意捐款)!")
                print(f"最终用户状态: {usr_da_next}")
                # 统计成功对话
                successful_dialogs += 1
                dialog_turn_counts.append(t + 1)
                successful_turn_counts.append(t + 1)
                total_turns += (t + 1)
                break
        else:
            # 如果循环正常结束（达到最大轮数），记录为未成功
            print(f"\n❌ 对话 {did} 达到最大轮数 ({max_step} 轮) 用户未同意捐款")
            dialog_turn_counts.append(max_step)
            total_turns += max_step
                
        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)
        num_done += 1
        pbar.update(1)

    pbar.close()

    # 计算并显示总运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n=== GDPZero P4G模拟实验完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))} ===")
    print(f"=== 总运行时间: {total_duration:.2f}秒 ({total_duration / 60:.2f}分钟) ===")
    if num_done > 0:
        print(f"=== 平均每个对话处理时间: {total_duration / num_done:.2f}秒 ===")

    # 计算并显示对话统计
    if num_done > 0:
        success_rate = (successful_dialogs / num_done) * 100
        average_turns = total_turns / num_done
        print(f"\n=== 对话统计结果 ===")
        print(f"总对话数: {num_done}")
        print(f"成功说服用户捐款的对话数: {successful_dialogs}")
        print(f"成功率: {success_rate:.1f}% ({successful_dialogs}/{num_done})")
        print(f"平均轮数: {average_turns:.1f}轮")
        print(f"各对话轮数分布: {dialog_turn_counts}")
        if successful_dialogs > 0:
            avg_successful_turns = sum(successful_turn_counts) / len(successful_turn_counts)
            print(f"成功对话的平均轮数: {avg_successful_turns:.1f}轮")
            print(f"成功对话的轮数分布: {successful_turn_counts}")

    print(f"\n所有目标对话模拟完成。共处理 {num_done} 个对话。总记录数 {len(output)}")
    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default=r"/Users/joey/Desktop/GDPZero-master/outputs/gdpzero_p4g_simulation.pkl",
                        help='output file')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo", "gpt-3.5-turbo-0613",
                                 "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat"],
                        help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1,
                        help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0,
                        help='initial Q value for uninitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=100, help='number of dialogs to test MCTS on')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    cmd_args = parser.parse_args()
    print("保存结果到", cmd_args.output)

    main(cmd_args)
