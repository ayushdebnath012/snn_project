import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    Penny-jumping market maker v4.

    Improvements over v3 (+2094 PnL):
    1. Asymmetric skew: position-reducing side always stays tight at penny-jump,
       position-building side widens. Captures more fills when directional.
    2. Lower base skew (0.10): never pushes quotes worse than penny-jump
       until extreme positions.
    3. Taker takes AT fair value (not just beyond), catching 8 extra EMERALD 
       trades at 10000.
    4. For TOMATOES, always undercuts whatever is best in book,
       even if another trader is already penny-jumping.
    """

    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}

    EMERALD_FAIR = 10000
    TOMATO_EMA_ALPHA = 0.35

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

    def _trade(self, state: TradingState, product: str, fair: float, data: dict) -> list[Order]:
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []

        # ========== 1) AGGRESSIVE TAKE ==========
        # Buy any ask <= fair, sell any bid >= fair
        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px <= fair:
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px >= fair:
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break

        # ========== 2) PENNY-JUMP WITH ASYMMETRIC SKEW ==========
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        fair_floor = math.floor(fair)
        fair_ceil = math.ceil(fair)

        # Determine penny-jump prices (undercut book by 1)
        pj_bid = (best_bid + 1) if best_bid is not None else (fair_floor - 1)
        pj_ask = (best_ask - 1) if best_ask is not None else (fair_ceil + 1)

        # Asymmetric skew:
        #   When LONG (pos>0):  keep ASK tight (to sell), push BID down (less buying)
        #   When SHORT (pos<0): keep BID tight (to buy), push ASK up (less selling)
        skew_amount = abs(pos) * 0.10

        if pos > 0:
            # Long: widen bid, keep ask tight
            our_bid = min(pj_bid, fair_floor - 1 - math.floor(skew_amount))
            our_ask = max(pj_ask, fair_ceil + 1)
        elif pos < 0:
            # Short: keep bid tight, widen ask
            our_bid = min(pj_bid, fair_floor - 1)
            our_ask = max(pj_ask, fair_ceil + 1 + math.ceil(skew_amount))
        else:
            our_bid = min(pj_bid, fair_floor - 1)
            our_ask = max(pj_ask, fair_ceil + 1)

        # Safety: never cross
        if our_bid >= our_ask:
            our_bid = fair_floor - 1
            our_ask = fair_ceil + 1

        # Primary layer — full remaining capacity
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
