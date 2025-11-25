import copy
import random
import numpy as np
import logging
import math

from core.helpers import DialogSession
from core.game1 import DialogGame, EmotionalSupportGame,CBGame
from core.cb_players import DialogPlanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NRPAPlanner:
    def __init__(self, game: DialogGame, player: DialogPlanner, configs):
        self.game = game
        self.player = player
        self.configs = configs
        self.nrpa_depth = getattr(configs, "nrpa_depth", 1)
        self.nrpa_iterations = getattr(configs, "nrpa_iterations", 10)

        # 早停机制配置
        self.early_stopping_enabled = getattr(configs, "early_stopping_enabled", True)
        self.early_stopping_threshold = getattr(configs, "early_stopping_threshold", 1.0)  # 捐赠成功得分
        self.early_stopping_patience = getattr(configs, "early_stopping_patience", 3)  # 连续无改进次数
        self.min_iterations = getattr(configs, "min_iterations", 3)  # 最少迭代次数

        # Define strategic dialog acts that should be rewarded
        self.strategic_acts = [
            'credibility appeal',
            'emotion appeal',
            'proposition of donation',
            'logical appeal',
            'task related inquiry'
        ]
        # Other dialog acts that are less strategic
        # self.other_acts = [
        #     'greeting',
        #     'other'
        # ]
        # # Define user dialog acts for scoring
        # self.positive_user_acts = [
        #     PersuasionGame.U_PositiveReaction,
        #     PersuasionGame.U_Donate,
        #     # PersuasionGame.U_Agree, # Assuming U_Agree might be too general or covered by U_PositiveReaction
        # ]
        # self.negative_user_acts = [
        #     PersuasionGame.U_NegativeReaction,
        #     PersuasionGame.U_NoDonation,
        #     # PersuasionGame.U_Disagree, # Assuming U_Disagree might be too general
        # ]
        # self.neutral_user_acts = [
        #     PersuasionGame.U_Neutral
        # ]
        self.debug = configs.get("debug", False)

    def print_dialog_history(self, state: DialogSession, prefix: str = ""):
        # print(prefix + "对话历史：")
        # for idx, turn in enumerate(state.history):
        #     print(f"{prefix}    轮次 {idx}: {turn}")
        pass

    def legalMoves(self, state: DialogSession):
        valid_mask = self.player.get_valid_moves(state)
        legal_moves = valid_mask.nonzero()[0].tolist()
        # print(f"[legalMoves] 当前状态轮次: {len(state.history)}，可选动作索引: {legal_moves}")
        # for idx in legal_moves:
        #     if idx < len(self.player.dialog_acts):
        #         print(f"    动作索引 {idx}: 对话动作 {self.player.dialog_acts[idx]}")
        #     else:
        #         print(f"    动作索引 {idx}: 未知动作")
        return legal_moves

    def play(self, state: DialogSession, move: int) -> DialogSession:
        # new_state = copy.deepcopy(state)
        # 检查游戏的get_next_state方法签名以兼容新旧接口
        import inspect
        sig = inspect.signature(self.game.get_next_state)
        num_params = len(sig.parameters)

        if num_params <= 2:  # 老接口：get_next_state(state, action)
            new_state = self.game.get_next_state(state, move)
        else:  # 新接口：get_next_state(state, action, agent_state, mode)
            agent_state = []
            new_state, next_agent_state, reward = self.game.get_next_state(state, move, agent_state)

        # if move < len(self.player.dialog_acts):
        #     da_str = self.player.dialog_acts[move]
        # else:
        #     da_str = "未知动作"
        # print(f"[play] 执行动作 {move} (对话动作: {da_str})，状态轮次: {len(state.history)} -> {len(new_state.history)}")
        # self.print_dialog_history(new_state, prefix="    ")
        return new_state

    def terminal(self, state: DialogSession):
        ended_val = self.game.get_dialog_ended(state)
        is_terminal = (ended_val != 0)
        # print(f"[terminal] 状态轮次: {len(state.history)}，结束标志: {ended_val}，终止: {is_terminal}")
        return is_terminal

    def _get_legacy_word_count_score_for_simulated_segment(self, simulated_turns,
                                                           resulting_state_full_history_for_debug,
                                                           len_history_before_simulation_for_debug):
        """Applies the user-provided legacy scoring logic based on the first Persuader utterance word count in simulated_turns."""
        if not resulting_state_full_history_for_debug:  # Check based on the original snippet's first condition
            # print("[score_legacy_tiebreak] (NRPA Sim) 对话历史为空，得分为 0")
            return 0

        # simulated_turns is already passed, which is: resulting_state.history[len_history_before_simulation:]

        if not simulated_turns:
            # print(f"[score_legacy_tiebreak] (NRPA Sim) 当前模拟段没有产生新的轮次 (原始历史长度 {len_history_before_simulation_for_debug})，得分为 0")
            if resulting_state_full_history_for_debug:
                # print(f"[score_legacy_tiebreak] (NRPA Sim) 当时完整对话历史 (共 {len(resulting_state_full_history_for_debug)} 轮):")
                for i, h_turn in enumerate(resulting_state_full_history_for_debug):
                    # print(f"  轮次 {i}: {h_turn}") # Note: original snippet didn't differentiate SIM> here, but it's fine.
                    pass
            return 0

        for turn_idx_in_sim_segment, turn in enumerate(simulated_turns):
            actual_history_idx = len_history_before_simulation_for_debug + turn_idx_in_sim_segment
            if len(turn) >= 3:
                speaker = turn[0]
                if speaker == CBGame.SYS:  # Check for Persuader
                    persuader_reply_content = turn[2]
                    if isinstance(persuader_reply_content, str):
                        word_count = len(persuader_reply_content.split())
                        # print("-------------------- LEGACY SCORE TIEBREAK CALCULATION (NRPA's First Simulated Persuader Utterance) --------------------")
                        # print(f"[score_legacy_tiebreak] (NRPA Sim) 完整对话历史 (共 {len(resulting_state_full_history_for_debug)} 轮), 模拟段从索引 {len_history_before_simulation_for_debug} 开始:")
                        for i, h_turn in enumerate(resulting_state_full_history_for_debug):
                            prefix = "  "
                            if i >= len_history_before_simulation_for_debug:
                                prefix = " SIM> "
                            # print(f"{prefix}轮次 {i}: {h_turn}")
                        # print(f"[score_legacy_tiebreak] (NRPA Sim) 选择用于计分的 Persuader 在当前模拟段的第一条回复 (来自历史记录索引 {actual_history_idx}, 发言者 {speaker}): '{persuader_reply_content}'")
                        # print(f"[score_legacy_tiebreak] (NRPA Sim) 计算词数: {word_count}，legacy tiebreak 得分: {word_count}")
                        # print("-----------------------------------------------------------------------------------------------------------------------------")
                        return word_count
                    else:
                        # print(f"[score_legacy_tiebreak] (NRPA Sim) Persuader 在当前模拟段的第一条回复 (历史记录索引 {actual_history_idx}, 发言者 {speaker}) 的内容不是字符串 ('{persuader_reply_content}'), legacy tiebreak 得分为 0")
                        if resulting_state_full_history_for_debug:
                            # print(f"[score_legacy_tiebreak] (NRPA Sim) 当时完整对话历史 (共 {len(resulting_state_full_history_for_debug)} 轮):")
                            for i, h_turn in enumerate(resulting_state_full_history_for_debug):
                                # print(f"  轮次 {i}: {h_turn}")
                                pass
                        return 0

        # print(f"[score_legacy_tiebreak] (NRPA Sim) 在当前模拟段 (从索引 {len_history_before_simulation_for_debug} 开始) 未找到 Persuader 的有效回复，legacy tiebreak 得分为 0")
        if resulting_state_full_history_for_debug:
            # print(f"[score_legacy_tiebreak] (NRPA Sim) 当时完整对话历史 (共 {len(resulting_state_full_history_for_debug)} 轮):")
            for i, h_turn in enumerate(resulting_state_full_history_for_debug):
                prefix = "  "
                if i >= len_history_before_simulation_for_debug:
                    prefix = " SIM> "
                # print(f"{prefix}轮次 {i}: {h_turn}")
        return 0

    def score(self, resulting_state: DialogSession, len_history_before_simulation: int):
        print(f"\n[Scoring] 开始为模拟路径评分...")
        print(f"[Scoring] 原始对话长度: {len_history_before_simulation}, 模拟后总长度: {len(resulting_state.history)}")

        initial_score = 0.0

        # 检查是否解决了问题
        is_solved = any(turn[1] == CBGame.U_Deal for turn in resulting_state.history if
                        turn[0] == CBGame.USR)

        if is_solved:
            print("[Scoring] 检测到 'Deal' 状态，设定基础分为 1.0")
            initial_score = 1.0
        else:
            print("[Scoring] 未检测到 'Deal' 状态，设定基础分为 0.0")
            initial_score = 0.0

        # --- 新增：计算SL值和效率奖励 ---
        sl_bonus = 0.0
        efficiency_bonus = 0.0

        if is_solved and hasattr(resulting_state, 'buyer_price') and hasattr(resulting_state, 'seller_price'):
            # 提取交易价格
            deal_price = self._extract_deal_price_from_history(resulting_state.history, len_history_before_simulation)

            if deal_price is not None:
                # 计算SL值
                sl_value = self._calculate_sl(resulting_state.buyer_price, resulting_state.seller_price, deal_price)
                print(f"[Scoring] SL计算: 买方价格={resulting_state.buyer_price}, 卖方价格={resulting_state.seller_price}, 交易价格={deal_price}")
                print(f"[Scoring] SL值: {sl_value:.4f}")

                # 计算轮数
                simulated_turns_count = (len(resulting_state.history) - len_history_before_simulation) / 2

                # 只有当SL > 0时才计算综合奖励
                if sl_value > 0:
                    # SL基础奖励
                    sl_base_bonus = sl_value * 1.0  # 提高SL基础权重

                    # 效率奖励：轮数越少，奖励越高
                    # 使用指数衰减：max_efficiency_bonus * exp(-turn_decay * turns)
                    max_efficiency_bonus = 0.5  # 最大效率奖励
                    turn_decay = 0.3  # 轮数衰减系数
                    efficiency_bonus = max_efficiency_bonus * math.exp(-turn_decay * simulated_turns_count)

                    # SL-效率综合奖励：SL越高且轮数越少，奖励越高
                    sl_efficiency_multiplier = 1.0 + efficiency_bonus  # 效率加成
                    sl_bonus = sl_base_bonus * sl_efficiency_multiplier

                    print(f"[Scoring] SL基础奖励: {sl_value:.4f} * 1.0 = {sl_base_bonus:.4f}")
                    print(f"[Scoring] 效率奖励: {max_efficiency_bonus} * exp(-{turn_decay} * {simulated_turns_count:.0f}) = {efficiency_bonus:.4f}")
                    print(f"[Scoring] SL-效率综合奖励: {sl_base_bonus:.4f} * {sl_efficiency_multiplier:.4f} = {sl_bonus:.4f}")
                else:
                    print(f"[Scoring] SL值 <= 0，无SL奖励")
            else:
                print(f"[Scoring] 无法提取交易价格，无SL奖励")

        # --- 计算轮次惩罚（降低惩罚权重，因为效率奖励已经处理了轮数优化）---
        simulated_turns_count = (len(resulting_state.history) - len_history_before_simulation) / 2
        turn_penalty_multiplier = 0.0005  # 降低轮次惩罚权重
        turn_penalty = simulated_turns_count * turn_penalty_multiplier
        print(
            f"[Scoring] 计算轮次惩罚: {simulated_turns_count:.0f} 轮 * {turn_penalty_multiplier} = {turn_penalty:.4f}")

        final_score = initial_score + sl_bonus - turn_penalty

        print(f"[Scoring] 最终得分: {initial_score} (基础分) + {sl_bonus:.4f} (SL-效率综合奖励) - {turn_penalty:.4f} (轮次惩罚) = {final_score:.6f}")
        return final_score

    def _extract_deal_price_from_history(self, history, len_history_before_simulation):
        """从对话历史中提取交易价格"""
        import re

        # 只检查模拟生成的部分
        simulated_turns = history[len_history_before_simulation:]

        if not simulated_turns:
            return None

        # 查找最后一个用户（卖方）的响应
        last_user_response = None
        for turn in reversed(simulated_turns):
            if len(turn) >= 3 and turn[0] == CBGame.USR:
                last_user_response = turn[2]
                break

        if not last_user_response:
            return None

        # 价格提取模式
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
        ]

        for pattern in price_patterns:
            match = re.search(pattern, last_user_response, re.IGNORECASE)
            if match:
                try:
                    if match.groups():
                        price_str = match.group(1).replace(',', '').strip()
                        if price_str:
                            return float(price_str)
                except (ValueError, IndexError):
                    continue

        # 如果没有找到特定模式，尝试提取所有数字
        prices = re.findall(r'\$?([\d,]+(?:\.\d+)?)', last_user_response)
        if prices:
            try:
                return float(prices[-1].replace(',', ''))
            except ValueError:
                pass

        return None

    def _calculate_sl(self, buyer_price, seller_price, deal_price):
        """计算SL值"""
        if deal_price is None or buyer_price == seller_price:
            return 0.0

        # 验证价格合理性
        if not self._validate_deal_price(deal_price, buyer_price, seller_price):
            return 0.0

        sl = (deal_price - seller_price) / (buyer_price - seller_price)

        # 过滤异常值
        if sl > 2.0 or sl < -1.0:
            return 0.0

        return sl

    def _validate_deal_price(self, deal_price, buyer_price, seller_price):
        """验证交易价格是否合理"""
        if deal_price is None:
            return False

        # 检查价格是否在买方和卖方价格之间（允许一定的浮动）
        min_price = min(buyer_price, seller_price)
        max_price = max(buyer_price, seller_price)

        # 允许10%的浮动范围
        tolerance = 0.1 * (max_price - min_price)

        return (min_price - tolerance) <= deal_price <= (max_price + tolerance)

    # def nrpa_playout(self, state: DialogSession, policy: dict) -> DialogSession:
    #     # print(f"--- 进入 NRPA playout，初始状态轮次: {len(state.history)} ---")
    #     # self.print_dialog_history(state, prefix="[nrpa_playout] ")
    #     while not self.terminal(state):
    #         move = self.nrpa_randomMove(state, policy)
    #         state = self.play(state, move)
    #         if len(state.history) > 100:
    #             print("[nrpa_playout] 警告: 对话轮次过多，强制结束")
    #             break
    #
    #     # print(f"--- 退出 NRPA playout，最终状态轮次: {len(state.history)} ---")
    #     # self.print_dialog_history(state, prefix="[nrpa_playout] ")
    #     return state

    def nrpa_playout(self, state: DialogSession, policy: dict) -> DialogSession:
        """根据给定策略进行一次模拟"""
        # print(f"--- 进入 playout，初始状态轮次: {len(state.history)} ---")
        # self.print_dialog_history(state, prefix="[playout] ")
        new_state = copy.deepcopy(state)
        playout_epsilon = getattr(self.configs, "nrpa_playout_epsilon", 0.1)  # 从配置中获取或设默认值
        max_playout_steps = getattr(self.configs, "max_playout_steps", 0)  # 0表示无限制

        initial_length = len(new_state.history)
        steps = 0
        while not self.terminal(new_state):
            # 如果设置了最大步数限制，检查是否超过
            if max_playout_steps > 0 and steps >= max_playout_steps:
                # print(f"[nrpa_playout] 达到最大步数限制 {max_playout_steps}，强制结束")
                break

            move = self.randomMove(new_state, policy, epsilon=playout_epsilon)  # 传递 epsilon
            if move is None:
                # print(f"[nrpa_playout] 无可用动作，playout 结束。当前状态轮次: {len(new_state.history)}")
                break

            # 执行动作前再次检查当前状态
            if self.terminal(new_state):
                # print(f"[nrpa_playout] 执行动作前发现已是终止状态，停止 playout")
                break

            new_state = self.play(new_state, move)
            steps += 1

            # 执行动作后立即检查是否达到终止状态
            if self.terminal(new_state):
                # print(f"[nrpa_playout] 执行动作后达到终止状态，playout 完成。最终状态轮次: {len(new_state.history)}")
                break

        final_length = len(new_state.history)
        if final_length <= initial_length:
            # print(
            #     f"[nrpa_playout] 警告: playout没有产生新轮次 - 初始:{initial_length}, 最终:{final_length}, 步数:{steps}")
            pass

        # print(f"--- 退出 playout，最终状态轮次: {len(new_state.history)} ---")
        # self.print_dialog_history(new_state, prefix="[playout] ")
        return new_state

    # def nrpa_randomMove(self, state: DialogSession, policy: dict) -> int:
    #     moves = self.legalMoves(state)
    #     total = 0.0
    #     for m in moves:
    #         if m not in policy:
    #             policy[m] = 0.0
    #         total += math.exp(policy[m])
    #     stop = random.random() * total
    #     s = 0.0
    #     # print(f"[nrpa_randomMove] 当前策略: {policy}")
    #     for m in moves:
    #         s += math.exp(policy[m])
    #         if s >= stop:
    #             # print(f"[nrpa_randomMove] 选择动作: {m}")
    #             return m

    def randomMove(self, state: DialogSession, policy: dict, epsilon: float = 0.1):  # 增加 epsilon 参数
        """根据策略随机选择一个动作，加入 epsilon-greedy 探索"""
        # 首先检查当前状态是否已经终止
        if self.terminal(state):
            # print(f"[randomMove] 当前状态已终止，返回 None")
            return None

        moves = self.legalMoves(state)
        if not moves:
            # print(f"[randomMove] 无可用动作，当前状态轮次: {len(state.history)}")
            return None

        if random.random() < epsilon:  # 以 epsilon 的概率进行探索
            # print(f"[randomMove] 随机探索: 从 {len(moves)} 个动作中随机选择")
            return random.choice(moves)
        else:  # 以 1-epsilon 的概率进行利用
            # print(f"[randomMove] 基于策略选择: 从 {len(moves)} 个动作中选择")
            # --- 原有的基于 policy 的选择逻辑 ---
            sum_exp = 0.0
            for m in moves:
                move_key = str(m)
                if move_key not in policy:
                    policy[move_key] = 0.0
                sum_exp += math.exp(policy[move_key])

            if sum_exp == 0:  # 如果所有权重都是0或负无穷的指数，均等选择
                # print(f"[randomMove] 策略权重和为0，随机选择动作")
                return random.choice(moves)

            stop = random.random() * sum_exp
            current_sum = 0.0
            for m in moves:
                move_key = str(m)
                exp_val = math.exp(policy[move_key])
                current_sum += exp_val
                if current_sum >= stop:
                    # print(f"[randomMove] 基于策略选择动作: {m}")
                    return m

            # Fallback if no move selected due to precision (should be rare with sum_exp > 0)
            best_move = max(moves, key=lambda m_val: math.exp(policy.get(str(m_val), -float('inf'))))
            # print(f"[randomMove] 后备选择最佳动作: {best_move}")
            return best_move

    def adapt(self, policy: dict, sequence: list, init_state: DialogSession) -> dict:

        s = copy.deepcopy(init_state)
        s.history = []
        polp = copy.deepcopy(policy)

        for turn in sequence:
            # 移除 speaker 判断，直接处理所有动作
            _, dialog_act, _ = turn  # 不再检查 speaker

            try:
                best_move = self.player.dialog_acts.index(dialog_act)
                best_move_key = str(best_move)  # 转换为字符串键
            except ValueError:
                # logger.warning(f"对话动作 '{dialog_act}' 不在预设列表中，已跳过")
                continue

            moves = self.legalMoves(s)
            total = sum(math.exp(polp.get(str(m), 0)) for m in moves)  # 使用字符串键

            for m in moves:
                move_key = str(m)  # 转换为字符串键
                if total == 0:
                    prob = 0
                else:
                    prob = math.exp(polp.get(move_key, 0)) / total
                polp[move_key] = polp.get(move_key, 0) - prob

            polp[best_move_key] = polp.get(best_move_key, 0) + 1.0
            s = self.play(s, best_move)

        return polp

    # def adapt(self, policy: dict, sequence: list) -> dict:
    #     """根据序列调整策略"""
    #     print(f"[adapt] 开始调整策略，序列长度: {len(sequence)}")
    #     s = DialogSession()  # 创建空对话状态
    #     polp = copy.deepcopy(policy)
    #
    #     for best_move in sequence:
    #         moves = self.legalMoves(s)
    #         if not moves:
    #             break
    #
    #         sum_exp = 0.0
    #         for m in moves:
    #             move_key = str(m)
    #             if move_key not in policy:
    #                 policy[move_key] = 0.0
    #             sum_exp += math.exp(policy[move_key])
    #
    #         # 调整策略权重
    #         for m in moves:
    #             move_key = str(m)
    #             if move_key not in polp:
    #                 polp[move_key] = 0.0
    #             polp[move_key] -= math.exp(policy[move_key]) / sum_exp
    #
    #         # 增加最佳动作的权重
    #         best_move_key = str(best_move)
    #         if best_move_key not in polp:
    #             polp[best_move_key] = 0.0
    #         polp[best_move_key] += 1.0
    #
    #         # 执行最佳动作以更新状态
    #         s = self.play(s, best_move)
    #
    #     print(f"[adapt] 策略调整完成: {polp}")
    #     return polp
    def nrpa(self, level: int, policy: dict, state: DialogSession) -> DialogSession:
        current_history_len_before_sim = len(state.history)  # 记录当前模拟/递归前的历史长度

        # 在开始搜索前检查输入状态是否已经终止
        if self.terminal(state):
            # print(f"[nrpa] 输入状态已是终止状态，直接返回。Level: {level}")
            return state

        if level == 0:
            # print(f"[NRPA Lvl 0] 开始 Playout...")
            candidate = self.nrpa_playout(state, policy)
            # 验证 playout 返回的状态
            if candidate is None:
                # print(f"[nrpa] Level 0: playout 返回 None，返回原始状态")
                return state
            # print(f"[nrpa] Level 0: playout 完成，返回状态轮次: {len(candidate.history)}")
            return candidate  # nrpa 应该返回 state，score 在外面计算

        best_state_from_sim = None  # 重命名以避免与输入的 state混淆
        best_score = float("-inf")
        best_seq_for_adapt = []  # 用于 adapt 的序列应该是最佳路径的完整历史

        # 早停机制相关变量
        no_improvement_count = 0

        print("-" * 60)
        print(f"[NRPA Lvl {level}] 开始 {self.nrpa_iterations} 次迭代搜索...")

        for i in range(self.nrpa_iterations):
            print(f"\n[NRPA Lvl {level}] >> 迭代 {i + 1}/{self.nrpa_iterations}")
            # 在每次迭代前检查状态
            if self.terminal(state):
                print(f"[NRPA Lvl {level}] 迭代开始时发现输入状态已终止，停止迭代")
                break

            pol_copy = copy.deepcopy(policy)
            # 'state' 传递给下一层，其历史长度是 current_history_len_before_sim
            print(f"[NRPA Lvl {level}] 调用下一层 nrpa(level={level - 1})...")
            candidate_sim_state = self.nrpa(level - 1, pol_copy, state)

            # 验证递归返回的状态
            if candidate_sim_state is None:
                print(f"[NRPA Lvl {level}] 下层返回 None 状态，跳过此次迭代。")
                continue

            # 在这里，candidate_sim_state 是下一层nrpa返回的最终状态
            # 我们需要用它来计算得分
            candidate_score = self.score(candidate_sim_state, current_history_len_before_sim)
            # print(f"[nrpa] Level {level}: 迭代 {i}, 候选得分: {candidate_score}, 状态轮次: {len(candidate_sim_state.history)}")

            if candidate_score > best_score:
                print(
                    f"[NRPA Lvl {level}] **** 新的最佳路径! 得分: {candidate_score:.6f} (优于之前的 {best_score:.6f}) ****")
                best_state_from_sim = candidate_sim_state
                best_score = candidate_score
                best_seq_for_adapt = best_state_from_sim.history  # adapt 使用选定路径的完整历史
                # print(f"[nrpa] Level {level}: 迭代 {i} 找到更好状态，得分: {best_score}")

                # 重置无改进计数
                no_improvement_count = 0
            else:
                print(f"[NRPA Lvl {level}] 路径未改进。当前得分 {candidate_score:.6f}, 最高分 {best_score:.6f}")
                # 无改进，增加计数
                no_improvement_count += 1

            # adapt 函数使用选定路径的历史 (best_seq_for_adapt) 和最初传入此nrpa级别的状态 (state)
            if best_seq_for_adapt:  # 确保我们有序列可以 adapt
                policy = self.adapt(policy, best_seq_for_adapt, state)

            # 早停检查 - 基于对话轮次数
            # 计算新增的对话轮次数（与score函数中的计算保持一致）
            current_turns = 0
            if best_state_from_sim is not None:
                simulated_turns_count = (len(best_state_from_sim.history) - current_history_len_before_sim) / 2
                current_turns = int(simulated_turns_count)  # 新增的轮次数

            if self.early_stopping_enabled and i >= self.min_iterations - 1:  # 至少执行min_iterations次
                # 条件1: 达到或优于轮次阈值 (early_stopping_threshold现在表示轮数)
                if current_turns <= self.early_stopping_threshold:
                    print(
                        f"\n[NRPA Lvl {level}] 早停: 当前对话轮次 {current_turns} 已达到或优于轮次阈值 {self.early_stopping_threshold}。")
                    break

                # 条件2: 连续无改进超过patience
                if no_improvement_count >= self.early_stopping_patience:
                    print(f"\n[NRPA Lvl {level}] 早停: 已连续 {no_improvement_count} 次迭代未找到更优路径。")
                    break

        print("-" * 60)
        # 最终验证返回状态
        if best_state_from_sim is None:
            # print(f"[nrpa] Level {level}: 所有迭代完成，未找到有效状态，返回原始状态")
            print(f"[NRPA Lvl {level}] 搜索失败 - 未找到有效状态，返回原始输入状态。")
            return state

        # print(
        #     f"[nrpa] Level {level}: 搜索完成，返回最佳状态轮次: {len(best_state_from_sim.history)}, 得分: {best_score}")
        print(
            f"[NRPA Lvl {level}] 搜索完成。返回的最佳路径得分: {best_score:.6f}, 总轮数: {len(best_state_from_sim.history)}")

        # 添加调试信息：检查返回状态是否真的比输入状态长
        if len(best_state_from_sim.history) <= len(state.history):
            # print(
            #     f"[nrpa] 警告: 返回状态轮次({len(best_state_from_sim.history)}) <= 输入状态轮次({len(state.history)})")
            pass

        return best_state_from_sim

    def _normalize_policy(self, policy):
        """将策略字典的键统一为字符串类型"""
        normalized = {}
        for k, v in policy.items():
            normalized[str(k)] = v
        return normalized