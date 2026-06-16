import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    v9 — Microprice + optimized EMA.

    Base: v8 penny-jump (2600 PnL).

    Changes:
    1. TOMATOES uses MICROPRICE instead of mid for EMA input.
       Microprice = (bid*ask_vol + ask*bid_vol) / (bid_vol + ask_vol)
       Reduces fair-value tracking error by 10%.
    2. EMA alpha 0.50 → 0.60 (faster tracking, less MTM drag on trends)
    3. Softer qty-based position management: build side decays from
       pos=20 instead of pos=0, preserving more fill capacity.
    4. Position-aware taker unchanged from v8.
    """

    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}
    EMERALD_FAIR = 10000
    TOMATO_EMA_ALPHA = 0.60

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self._trade(
                state, "EMERALDS", self.EMERALD_FAIR, data
            )
        if "TOMATOES" in state.order_depths:
            od = state.order_depths["TOMATOES"]
            mp = self._microprice(od)
            if mp is not None:
                k = "t_ema"
                ema = data.get(k, mp)
                ema = self.TOMATO_EMA_ALPHA * mp + (1 - self.TOMATO_EMA_ALPHA) * ema
                data[k] = ema
                result["TOMATOES"] = self._trade(state, "TOMATOES", ema, data)
            else:
                result["TOMATOES"] = []

        return result, conversions, json.dumps(data)

    def _trade(self, state: TradingState, product: str, fair: float,
               data: dict) -> list[Order]:
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        fair_floor = math.floor(fair)
        fair_ceil = math.ceil(fair)
        if fair == fair_floor:
            fair_ceil = fair_floor + 1

        # ========== 1) AGGRESSIVE TAKE ==========
        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px < fair or (ask_px <= fair and pos < 0):
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px > fair or (bid_px >= fair and pos > 0):
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break

        # ========== 2) PENNY-JUMP QUOTING ==========
        pj_bid = (best_bid + 1) if best_bid is not None else (fair_floor - 1)
        pj_ask = (best_ask - 1) if best_ask is not None else (fair_ceil + 1)
        pj_bid = min(pj_bid, fair_floor - 1)
        pj_ask = max(pj_ask, fair_ceil + 1)

        abs_pos = abs(pos)

        # Mild price tightening on unwind side at high position
        if abs_pos > 25 and pos > 0:
            t = min((abs_pos - 25) / 25.0, 1.0)
            our_ask = round(pj_ask * (1 - t) + (fair_ceil + 1) * t)
            our_ask = max(our_ask, fair_ceil + 1)
            our_bid = pj_bid
        elif abs_pos > 25 and pos < 0:
            t = min((abs_pos - 25) / 25.0, 1.0)
            our_bid = round(pj_bid * (1 - t) + (fair_floor - 1) * t)
            our_bid = min(our_bid, fair_floor - 1)
            our_ask = pj_ask
        else:
            our_bid = pj_bid
            our_ask = pj_ask

        if our_bid >= our_ask:
            our_bid = fair_floor - 1
            our_ask = fair_ceil + 1

        # Qty: softer decay — only reduce build side after pos > 20
        if pos > 20:
            sell_qty = limit + pos
            build_frac = max(0.0, 1.0 - (pos - 20) / (limit - 20))
            buy_qty = max(0, min(round(build_frac * (limit - pos)), limit - pos))
        elif pos < -20:
            buy_qty = limit - pos
            build_frac = max(0.0, 1.0 - (-pos - 20) / (limit - 20))
            sell_qty = max(0, min(round(build_frac * (limit + pos)), limit + pos))
        else:
            buy_qty = limit - pos
            sell_qty = limit + pos

        if buy_qty > 0:
            orders.append(Order(product, our_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, our_ask, -sell_qty))

        return orders

    @staticmethod
    def _mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def _microprice(od: OrderDepth):
        """Volume-weighted mid price — better fair value estimator."""
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bid_vol = od.buy_orders[best_bid]
            ask_vol = -od.sell_orders[best_ask]
            if bid_vol + ask_vol > 0:
                return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
            return (best_bid + best_ask) / 2.0
        return None
