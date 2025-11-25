import numpy as np
import logging
import pickle
import argparse
import os
import time
import re

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
from core.cb_players import (
    BuyerModel, SellerModel, CBSystemPlanner,
    BuyerChatModel, SellerChatModel, CBChatSystemPlanner
)
from core.game1 import EmotionalSupportGame, CBGame
from core.game import PersuasionGame
from core.helpers import DialogSession, CBDialogSession
from utils.utils import dotdict
from utils.prompt_examples import ESConv_EXP_DIALOG, CB_EXP_DIALOG

from core.sr_nrpa_cb import NRPAPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 为本地模型设置日志级别
logging.getLogger('core.gen_models').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def extract_deal_price_from_llm_response(llm_response_text):
    """从LLM的交易判断响应中提取价格"""
    # 查找 "deal at [price]" 格式的价格
    deal_patterns = [
        r'deal at \$?([\d,]+\.?\d*)',
        r'reached a deal at \$?([\d,]+\.?\d*)',
        r'agreed on \$?([\d,]+\.?\d*)',
        r'settled on \$?([\d,]+\.?\d*)',
        r'price of \$?([\d,]+\.?\d*)',
        r'for \$?([\d,]+\.?\d*)',
    ]

    for pattern in deal_patterns:
        match = re.search(pattern, llm_response_text, re.IGNORECASE)
        if match:
            try:
                price_str = match.group(1).replace(',', '').strip()
                if price_str:
                    return float(price_str)
            except (ValueError, IndexError):
                continue

    # 如果没有找到特定格式，尝试提取所有数字
    prices = re.findall(r"[-+]?\d*\.?\d+", llm_response_text.replace(",", ""))
    if prices:
        try:
            return float(prices[0])
        except ValueError:
            pass
    return None


def extract_deal_price_with_llm(context, state, backbone_model=None):
    """使用LLM来判断交易并提取价格

    Args:
        context: 对话上下文
        state: CBDialogSession 对象
        backbone_model: 用于价格提取的语言模型

    Returns:
        tuple: (deal_price, is_deal_reached)
    """
    if not backbone_model:
        return None, False

    # 构建提示信息
    messages = [
        {"role": "system",
         "content": "Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."},
        {"role": "user",
         "content": f"Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price. You can only reply with one of the following formats:\n- 'They have reached a deal at $[price].' (if deal is reached)\n- 'They have not reached a deal.' (if no deal)\n\nThe following is the conversation:\n{context}\n\nQuestion: Have they reached a deal? Answer:"}
    ]

    try:
        # 使用LLM判断交易
        eval_args = {
            "max_new_tokens": 32,
            "temperature": 0.0,
            "num_return_sequences": 1,
        }
        data = backbone_model.chat_generate(messages, **eval_args)

        if data and len(data) > 0:
            response = data[0]['generated_text'].strip()

            # 判断是否达成交易
            if 'have not' in response.lower() or 'no deal' in response.lower():
                return None, False
            elif 'have reached' in response.lower() or 'deal at' in response.lower():
                # 提取价格
                deal_price = extract_deal_price_from_llm_response(response)
                return deal_price, True
    except Exception as e:
        print(f"LLM价格提取失败: {e}")

    return None, False


def extract_deal_price(usr_resp, context, state=None):
    """从卖方响应和上下文中提取交易价格

    Args:
        usr_resp: 卖方的响应文本
        context: 对话上下文
        state: CBDialogSession 对象，包含买方和卖方的价格信息

    Returns:
        float: 提取到的交易价格，如果没有提取到则返回None
    """
    # 如果有 state 对象，可以利用其中的价格信息进行更智能的提取
    if state and hasattr(state, 'buyer_price') and hasattr(state, 'seller_price'):
        buyer_price = state.buyer_price
        seller_price = state.seller_price
        price_range = (min(buyer_price, seller_price), max(buyer_price, seller_price))
    else:
        price_range = (0, 1000000)  # 默认价格范围

    # 更精确的价格提取模式
    price_patterns = [
        r'agree to the proposed price of \$?([\d,]+\.?\d*)',
        r'I agree to.*?price of \$?([\d,]+\.?\d*)',
        r'agree to.*?\$?([\d,]+\.?\d*)',
        r'\$?([\d,]+\.?\d*) it is',
        r'deal at \$?([\d,]+\.?\d*)',
        r'We have a deal at \$?([\d,]+\.?\d*)',
        r'I can accept \$?([\d,]+\.?\d*)',
        r'accept \$?([\d,]+\.?\d*)',
        r"I'll accept \$?([\d,]+\.?\d*)",
        r"Let's go with \$?([\d,]+\.?\d*)",
        r'go with \$?([\d,]+\.?\d*)',
        r'price of \$?([\d,]+\.?\d*)',
        r'\$?([\d,]+\.?\d*) sounds like a fair compromise',
        r'\$?([\d,]+\.?\d*) sounds fair',
        r'settle on \$?([\d,]+\.?\d*)',
        r"Let's settle on \$?([\d,]+\.?\d*)",
        r'I can accept your offer of \$?([\d,]+\.?\d*)',
        r'accept your offer of \$?([\d,]+\.?\d*)',
        r'Deal! \$?([\d,]+\.?\d*)',
        r'Sold for \$?([\d,]+\.?\d*)',
        r'final price.*?\$?([\d,]+\.?\d*)',
        r'at \$?([\d,]+\.?\d*)',
        r'for \$?([\d,]+\.?\d*)',
        r'sell it for \$?([\d,]+\.?\d*)',
        r'buy it for \$?([\d,]+\.?\d*)',
    ]

    # 首先在卖方响应中查找
    for pattern in price_patterns:
        match = re.search(pattern, usr_resp, re.IGNORECASE)
        if match:
            try:
                if match.groups():
                    price_str = match.group(1).replace(',', '').strip()
                    if price_str:
                        price = float(price_str)
                        # 检查价格是否在合理范围内
                        if price_range[0] <= price <= price_range[1]:
                            return price
            except (ValueError, IndexError):
                continue

    # 如果在卖方响应中没找到，从上下文中查找最后提到的价格
    all_prices = re.findall(r'\$?([\d,]+(?:\.\d+)?)', context)
    if all_prices:
        for price_str in reversed(all_prices):  # 从后往前找
            try:
                price = float(price_str.replace(',', ''))
                # 检查价格是否在合理范围内
                if price_range[0] <= price <= price_range[1]:
                    return price
            except ValueError:
                continue

    return None


def validate_deal_price(deal_price, buyer_price, seller_price):
    """验证交易价格是否合理

    Args:
        deal_price: 提取的交易价格
        buyer_price: 买方出价
        seller_price: 卖方出价

    Returns:
        bool: 价格是否合理
    """
    if deal_price is None:
        return False

    # 检查价格是否在买方和卖方价格之间（允许一定的浮动）
    min_price = min(buyer_price, seller_price)
    max_price = max(buyer_price, seller_price)

    # 允许10%的浮动范围
    tolerance = 0.1 * (max_price - min_price)

    return (min_price - tolerance) <= deal_price <= (max_price + tolerance)


def calculate_sl_for_dialog(buyer_price, seller_price, deal_price):
    """计算单个对话的SL值
    SL = (deal_price - seller_price) / (buyer_price - seller_price)
    如果没有达成交易或价格无效，返回0
    """
    if deal_price is None or buyer_price == seller_price:
        return 0.0

    # 验证价格合理性
    if not validate_deal_price(deal_price, buyer_price, seller_price):
        print(f"⚠️  价格异常: 交易价格={deal_price}, 买方价格={buyer_price}, 卖方价格={seller_price}")
        return 0.0

    sl = (deal_price - seller_price) / (buyer_price - seller_price)

    # 过滤异常值
    if sl > 2.0 or sl < -1.0:
        print(f"⚠️  SL值异常: {sl:.4f}")
        return 0.0

    return sl


def main(cmd_args):
    # 记录总体开始时间
    total_start_time = time.time()
    print(f"=== NRPA 实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))} ===")

    system_name = CBGame.SYS
    user_name = CBGame.USR

    exp_1 = DialogSession(system_name, user_name).from_history(CB_EXP_DIALOG)

    game_ontology = CBGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']

    if cmd_args.llm == 'code-davinci-002':
        backbone_model = OpenAIModel(cmd_args.llm)
        SysModel = BuyerModel
        UsrModel = SellerModel
        SysPlanner = CBSystemPlanner
    elif cmd_args.llm in ['gpt-3.5-turbo']:
        backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
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
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
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
    elif cmd_args.llm == 'qwen-plus':
        backbone_model = QwenPlusChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'qwen-turbo':
        backbone_model = QwenTurboChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'qwen3-0.6b':
        backbone_model = QwenMaxChatModel(cmd_args.gen_sentences)
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'local-openai':
        # 通用本地 OpenAI 兼容模型
        backbone_model = LocalOpenAIChatModel(
            model_name=getattr(cmd_args, 'local_model_name', 'xxx'),
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1'),
            gen_sentences=cmd_args.gen_sentences
        )
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'local-qwen':
        # 本地 Qwen 模型
        backbone_model = LocalQwenChatModel(
            gen_sentences=cmd_args.gen_sentences,
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1')
        )
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'local-llama':
        # 本地 Llama 模型
        backbone_model = LocalLlamaChatModel(
            gen_sentences=cmd_args.gen_sentences,
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1')
        )
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
    elif cmd_args.llm == 'local-chatglm':
        # 本地 ChatGLM 模型
        backbone_model = LocalChatGLMChatModel(
            gen_sentences=cmd_args.gen_sentences,
            base_url=getattr(cmd_args, 'local_base_url', 'http://localhost:6006/v1')
        )
        SysModel = BuyerChatModel
        UsrModel = SellerChatModel
        SysPlanner = CBChatSystemPlanner
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

    game = CBGame(system, user, planner, zero_shot=False)
    print(f"使用模型: {cmd_args.llm}")
    print(f"系统对话行为: {system.dialog_acts}")
    print(f"用户对话行为: {user.dialog_acts}")

    import json
    all_dialogs = {}
    with open(r"D:\GDPZero-master\data\cb-test.txt", "r", encoding="utf-8") as f:
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

        # CB对话场景的数据结构
        item_name = dialog.get("item_name", "unknown")
        buyer_item_description = dialog.get("buyer_item_description", "")
        buyer_price = dialog.get("buyer_price", 0)
        seller_item_description = dialog.get("seller_item_description", "")
        seller_price = dialog.get("seller_price", 0)

        logger.info("evaluating dialog item: {}".format(item_name))
        initial_state = game.init_dialog(
            item_name,
            buyer_item_description,
            buyer_price,
            seller_item_description,
            seller_price
        )

        # 按照您提供的初始对话场景
        sys_role = CBGame.SYS
        usr_role = CBGame.USR
        history = [(sys_role, CBGame.S_Inquire, "Hi, how much is the %s?" % item_name),
                   (usr_role, CBGame.U_No_deal,
                    "Hi, this is a good %s and its price is %s." % (item_name, seller_price))]
        initial_state.history = history  # 直接覆盖历史

        # 从历史记录中获取上下文用于打印
        sys_utt = history[0][2]
        usr_utt = history[1][2]
        end_condition = CBGame.U_Deal

        context = f"""
        {sys_role}: {sys_utt}
        {usr_role}: {usr_utt}
        """
        initial_context = context.replace('\t', '').strip()
        print(f"\n=== 开始模拟对话 {did} ===")
        print(f"初始对话上下文:\n{initial_context}\n" + "=" * 50)

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
                print(f"Buyer: [{sys_da}] {sys_resp}")
                print(f"Seller: [{usr_da}] {usr_resp}")

                current_context += f"\nBuyer: {sys_resp}\nSeller: {usr_resp}"

                cmp_data = {
                    'did': did,
                    'turn': turn_count_in_sim,
                    'context': current_context.strip(),
                    'new_resp': sys_resp,
                    'new_da': sys_da,
                    'usr_resp': usr_resp,
                    'usr_da': usr_da,
                    "debug": {
                        "nrpa_iterations": nrpa_args.nrpa_iterations,
                        "nrpa_depth": nrpa_args.nrpa_depth,
                        "nrpa_search_time": nrpa_duration,
                    }
                }
                output.append(cmp_data)

                if usr_da == CBGame.U_Deal:
                    is_solved = True
                    break

            print("-" * 50)

            # 计算SL值
            deal_price = None
            sl_value = 0.0

            if is_solved:
                # 首先尝试使用LLM提取价格
                deal_price, llm_deal_confirmed = extract_deal_price_with_llm(current_context, final_state,
                                                                             backbone_model)

                # 如果LLM提取失败，使用传统方法
                if deal_price is None:
                    final_usr_resp = simulated_turns[-1][2] if len(simulated_turns) >= 2 else ""
                    deal_price = extract_deal_price(final_usr_resp, current_context, final_state)

                # 计算SL值
                sl_value = calculate_sl_for_dialog(initial_state.buyer_price, initial_state.seller_price, deal_price)

                price_source = "LLM提取" if llm_deal_confirmed else "正则提取"
                print(f"\n🎉 对话 {did} 在第 {turn_count_in_sim} 轮结束 (交易成功)!")
                print(
                    f"📊 SL计算: 买方价格={initial_state.buyer_price}, 卖方价格={initial_state.seller_price}, 交易价格={deal_price} ({price_source})")
                print(f"📈 SL值: {sl_value:.4f}")

                successful_dialogs += 1
                successful_turn_counts.append(turn_count_in_sim)
            else:
                print(f"\n❌ 对话 {did} 模拟结束时未达成交易 (共 {turn_count_in_sim} 轮)")
                print(
                    f"📊 SL计算: 买方价格={initial_state.buyer_price}, 卖方价格={initial_state.seller_price}, 交易价格=无")
                print(f"📈 SL值: {sl_value:.4f} (未达成交易)")

            # 在最后一条记录中添加SL相关信息
            if output:
                output[-1]['sl_value'] = sl_value
                output[-1]['deal_price'] = deal_price
                output[-1]['buyer_price'] = initial_state.buyer_price
                output[-1]['seller_price'] = initial_state.seller_price
                output[-1]['deal_reached'] = is_solved
                if is_solved:
                    output[-1]['price_extraction_method'] = price_source

            dialog_turn_counts.append(turn_count_in_sim)
            total_turns += turn_count_in_sim

        else:
            print("警告: NRPA未能生成有效对话。")
            dialog_turn_counts.append(0)
            # 即使没有生成对话，也要记录SL信息
            deal_price = None
            sl_value = 0.0
            print(f"📊 SL计算: 买方价格={initial_state.buyer_price}, 卖方价格={initial_state.seller_price}, 交易价格=无")
            print(f"📈 SL值: {sl_value:.4f} (无对话生成)")

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
        print(f"成功达成交易的对话数: {successful_dialogs}")
        print(f"成功率: {success_rate:.1f}% ({successful_dialogs}/{num_done})")
        print(f"平均轮数: {average_turns:.1f}轮")
        print(f"各对话轮数分布: {dialog_turn_counts}")
        if successful_dialogs > 0:
            avg_successful_turns = sum(successful_turn_counts) / len(successful_turn_counts)
            print(f"成功交易的平均轮数: {avg_successful_turns:.1f}轮")
            print(f"成功交易的轮数分布: {successful_turn_counts}")

    print(f"\n所有对话处理完成。共处理 {len(processed_dialogs)} 个不同对话ID。总记录数: {len(output)}")

    # 计算SL统计信息
    sl_values = []
    deal_count = 0
    total_deal_price = 0

    for item in output:
        if 'sl_value' in item:
            sl_values.append(item['sl_value'])
            if item.get('deal_reached', False) and item.get('deal_price') is not None:
                deal_count += 1
                total_deal_price += item['deal_price']

    if sl_values:
        print(f"\n=== SL统计结果 ===")
        print(f"总SL样本数: {len(sl_values)}")
        print(f"平均SL: {np.mean(sl_values):.4f}")
        print(f"SL标准差: {np.std(sl_values):.4f}")
        print(f"最小SL: {np.min(sl_values):.4f}")
        print(f"最大SL: {np.max(sl_values):.4f}")
        print(f"中位数SL: {np.median(sl_values):.4f}")

        # 统计分布
        zero_count = sum(1 for x in sl_values if abs(x) < 0.0001)
        positive_count = sum(1 for x in sl_values if x > 0.0001)
        negative_count = sum(1 for x in sl_values if x < -0.0001)

        print(f"SL=0的样本数: {zero_count} ({zero_count / len(sl_values) * 100:.1f}%)")
        print(f"SL>0的样本数: {positive_count} ({positive_count / len(sl_values) * 100:.1f}%)")
        print(f"SL<0的样本数: {negative_count} ({negative_count / len(sl_values) * 100:.1f}%)")

        if deal_count > 0:
            print(f"成功交易数: {deal_count}")
            print(f"平均交易价格: {total_deal_price / deal_count:.2f}")

    with open(cmd_args.output, "wb") as f:
        pickle.dump(output, f)
    print(f"最终结果已保存到: {cmd_args.output}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default=r"D:\GDPZero-master\outputs\gpt-4o-mini_nrpa_CB_test_sl.pkl",
                        help='output file path')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini",
                        choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt", "gpt-3.5-turbo-0613",
                                 "gpt-4-turbo-2024-04-09", "gpt-4o-mini-2024-07-18", "deepseek-chat", "gpt-4o-mini",
                                 "qwen2-7b-instruct", "qwen-plus", "qwen-turbo", "qwen3-0.6b",
                                 "local-openai", "local-qwen", "local-llama", "local-chatglm"],
                        help='LLM backbone model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences for LLM generation')
    parser.add_argument('--nrpa_depth', type=int, default=1, help='NRPA recursive search depth. 0 for pure playout.')
    parser.add_argument('--nrpa_iterations', type=int, default=5, help='Number of iterations per NRPA level')
    parser.add_argument('--num_dialogs', type=int, default=1000, help='Target number of dialogs to process')
    parser.add_argument('--nrpa_playout_epsilon', type=float, default=0,
                        help='Epsilon for epsilon-greedy exploration in NRPA playouts')
    parser.add_argument('--reduced_iterations', type=int, default=0,
                        help='Reduced number of iterations (overrides nrpa_iterations if > 0)')
    parser.add_argument('--max_playout_steps', type=int, default=10, help='Maximum playout steps (0 means unlimited)')
    parser.add_argument('--start_dialog', type=int, default=1, help='Start processing from dialog number (1-based)')
    parser.add_argument('--early_stopping_enabled', type=bool, default=True, help='Enable early stopping mechanism')
    parser.add_argument('--early_stopping_threshold', type=int, default=3,
                        help='Early stopping threshold (dialog turns)')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min_iterations', type=int, default=3, help='Minimum iterations')
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
