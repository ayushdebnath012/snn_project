import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    Penny-jumping market maker v5c.
    
    Core: penny-jump the bot spread to intercept all aggressive flow.
    
    Improvements over v4 (2200 PnL):
    1. TOMATO EMA alpha 0.50 — faster tracking of the ~50pt daily drift.
    2. Position-aware taker: when reducing position, accept 1 tick worse;
       when building position, require 1 tick better. Asymmetric edge.
    3. Graduated position skew on the UNWIND side only:
       - pos 0-25: pure penny-jump both sides
       - pos 25-40: tighten unwind side toward fair (more attractive fills)
       - pos 40+: quote unwind side AT fair to dump urgently
       Position-building side stays at penny-jump always.
    """

    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}
    EMERALD_FAIR = 10000
    TOMATO_EMA_ALPHA = 0.50

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
            mid = self._mid(od)
            if mid is not None:
                k = "t_ema"
                ema = data.get(k, mid)
                ema = self.TOMATO_EMA_ALPHA * mid + (1 - self.TOMATO_EMA_ALPHA) * ema
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

        # ========== 1) AGGRESSIVE TAKE ==========
        if pos > 0:
            buy_thresh = fair - 1
            sell_thresh = fair - 1
        elif pos < 0:
            buy_thresh = fair + 1
            sell_thresh = fair + 1
        else:
            buy_thresh = fair
            sell_thresh = fair

        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px <= buy_thresh:
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px >= sell_thresh:
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break

        # ========== 2) PENNY-JUMP QUOTING WITH GRADUATED SKEW ==========
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        fair_floor = math.floor(fair)
        fair_ceil = math.ceil(fair)
        if fair_floor == fair_ceil:
            fair_ceil += 1

        # Base penny-jump prices
        pj_bid = (best_bid + 1) if best_bid is not None else (fair_floor - 1)
        pj_ask = (best_ask - 1) if best_ask is not None else (fair_ceil + 1)

        # Clamp to not cross fair
        pj_bid = min(pj_bid, fair_floor - 1)
        pj_ask = max(pj_ask, fair_ceil + 1)

        # Graduated skew: tighten unwind side based on position magnitude
        abs_pos = abs(pos)
        if abs_pos <= 25:
            # Comfortable: pure penny-jump both sides
            our_bid = pj_bid
            our_ask = pj_ask
        else:
            # Interpolate unwind side from penny-jump toward fair±1
            # At pos=25: full penny-jump, at pos=50: at fair±1
            t = min((abs_pos - 25) / 25.0, 1.0)  # 0 to 1

            if pos > 0:
                # Long: tighten ask (sell side) toward fair+1
                target_ask = fair_ceil + 1
                our_ask = round(pj_ask * (1 - t) + target_ask * t)
                our_ask = max(our_ask, fair_ceil + 1)
                our_bid = pj_bid  # keep bid at penny-jump
            else:
                # Short: tighten bid (buy side) toward fair-1
                target_bid = fair_floor - 1
                our_bid = round(pj_bid * (1 - t) + target_bid * t)
                our_bid = min(our_bid, fair_floor - 1)
                our_ask = pj_ask  # keep ask at penny-jump

        # Safety
        if our_bid >= our_ask:
            our_bid = fair_floor - 1
            our_ask = fair_ceil + 1

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
