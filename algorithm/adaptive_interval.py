from collections import deque
import math

import numpy as np

from environment.config import AdaptiveIntervalConfig

class AdaptiveDecisionInterval:
    # 论文式 (12)–(18) 对应 rate_window 模式；load/random 是其他可选模式。
    # config.min_interval_s / max_interval_s: Δ_min / Δ_max。
    # config.adjustment_s: t_min；high/low_change_threshold: θ_h / θ_l。
    def __init__(
        self,
        config: AdaptiveIntervalConfig,
        initial_interval_s: float,
        seed: int = 0,
    ):
        config.validate(initial_interval_s)
        self.config = config
        self.initial_interval_s = float(initial_interval_s)
        self.seed = int(seed)

        self.changes = deque(maxlen=config.history_length)

        self.rate_changes = deque()

        self.history_window_s = config.history_length * self.initial_interval_s
        self.input_token_rate_floor = config.input_token_floor / self.initial_interval_s
        # Δ_stable
        self.required_stable_time_s = config.stable_cycles * self.initial_interval_s

        self.reset(0)

    def reset(self, arrived_input_tokens: int, episode: int = 0) -> None:
        self.interval_s = self.initial_interval_s
        self.previous_input_tokens = int(arrived_input_tokens)

        self.changes.clear()
        self.rate_changes.clear()

        self.stable_count = 0
        self.stable_duration_s = 0.0

        self.latest_change = 0.0
        self.window_change = 0.0
        self.history_covered_s = 0.0

        self.previous_input_token_rate = None

        if self.config.enabled and self.config.mode == "random":

            self.rng = np.random.default_rng([self.seed, int(episode)])
            self._sample_interval()

    def _sample_interval(self) -> float:
        choices = int(
            round(
                (self.config.max_interval_s - self.config.min_interval_s)
                / self.config.random_step_s
            )
        ) + 1
        index = int(self.rng.integers(choices))
        self.interval_s = (
            self.config.min_interval_s
            + index * self.config.random_step_s
        )
        return self.interval_s

    def _append_rate_change(self, change: float, duration_s: float) -> None:
        
        self.rate_changes.append([float(change), float(duration_s)])
        total = sum(item[1] for item in self.rate_changes)

        excess = total - self.history_window_s
        while excess > 1e-12 and self.rate_changes:
            oldest_change, oldest_duration = self.rate_changes[0]

            if oldest_duration <= excess + 1e-12:
                excess -= oldest_duration
                self.rate_changes.popleft()
            else:
                self.rate_changes[0][1] = oldest_duration - excess
                excess = 0.0

        self.history_covered_s = sum(
            item[1] for item in self.rate_changes
        )

        if self.history_covered_s > 0:
            # f̄_k 
            self.window_change = (
                sum(change_i * duration_i
                    for change_i, duration_i in self.rate_changes)
                / self.history_covered_s
            )
        else:
            self.window_change = 0.0

    def _update_rate_window(
        self,
        arrived_input_tokens: int,
        elapsed_s: float,
    ) -> float:
        #  AT_k
        # ρ_k 
        current_rate = float(arrived_input_tokens) / elapsed_s

        if self.previous_input_token_rate is None:
            # 第一个区间要保持初始间隔。
            self.previous_input_token_rate = current_rate
            self.latest_change = 0.0
            self.window_change = 0.0
            self.history_covered_s = 0.0
            self.stable_duration_s = 0.0
            self.stable_count = 0
            return self.interval_s

        previous_rate = self.previous_input_token_rate  
        self.previous_input_token_rate = current_rate

        
        denominator = max(
            current_rate,
            previous_rate,
            self.input_token_rate_floor,
        )

        if denominator > 0:
            # f_k
            self.latest_change = (
                abs(current_rate - previous_rate) / denominator
            )
        else:

            self.latest_change = 0.0

        self._append_rate_change(
            change=self.latest_change,
            duration_s=elapsed_s,
        )

        # F_k
        change_signal = max(
            self.latest_change,
            self.window_change,
        )

        if change_signal >= self.config.high_change_threshold:
            # 判定
            self.interval_s = max(
                self.config.min_interval_s,
                self.interval_s - self.config.adjustment_s,
            )
            self.stable_duration_s = 0.0
            self.stable_count = 0
            return self.interval_s

       
        history_ready = math.isclose(
            self.history_covered_s,
            self.history_window_s,
            rel_tol=0.0,
            abs_tol=1e-7,
        )

        if (
            history_ready
            and change_signal <= self.config.low_change_threshold
        ):
            # Δ_k 
            self.stable_duration_s += elapsed_s
            self.stable_count += 1

            if self.stable_duration_s >= self.required_stable_time_s:
                
                self.interval_s = min(
                    self.config.max_interval_s,
                    self.interval_s + self.config.adjustment_s,
                )
                # 重置 Δ_k
                self.stable_duration_s = 0.0
                self.stable_count = 0

            return self.interval_s

       
        self.stable_duration_s = 0.0
        self.stable_count = 0
        return self.interval_s

    def update(self, arrived_input_tokens: int, elapsed_s: float) -> float:
        if (
            not math.isfinite(elapsed_s)
            or elapsed_s <= 0
            or arrived_input_tokens < 0
        ):
            raise ValueError(
                "interval update needs positive elapsed time and "
                "non-negative arrived input tokens"
            )

        previous = self.previous_input_tokens
        self.previous_input_tokens = int(arrived_input_tokens)

        if not self.config.enabled:
            return self.interval_s

        if self.config.mode == "fixed":
            return self.interval_s

        if self.config.mode == "random":
            return self._sample_interval()

        if self.config.mode == "rate_window":
            return self._update_rate_window(
                arrived_input_tokens=arrived_input_tokens,
                elapsed_s=elapsed_s,
            )

        self.latest_change = (
            abs(arrived_input_tokens - previous)
            / max(
                arrived_input_tokens,
                previous,
                self.config.input_token_floor,
            )
            * self.initial_interval_s
            / elapsed_s
        )
        self.changes.append(self.latest_change)
        self.window_change = sum(self.changes) / len(self.changes)

        if (
            max(self.latest_change, self.window_change)
            >= self.config.high_change_threshold
        ):
            self.interval_s = max(
                self.config.min_interval_s,
                self.interval_s - self.config.adjustment_s,
            )
            self.stable_count = 0

        elif (
            len(self.changes) == self.config.history_length
            and max(self.latest_change, self.window_change)
            <= self.config.low_change_threshold
        ):
            self.stable_count += 1

            if self.stable_count >= self.config.stable_cycles:
                self.interval_s = min(
                    self.config.max_interval_s,
                    self.interval_s + self.config.adjustment_s,
                )
                self.stable_count = 0

        else:
            self.stable_count = 0

        return self.interval_s

