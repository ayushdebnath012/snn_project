import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    v8 — Back to penny-jump (v6 base = 2575), improved position mgmt.

    Fix: v6's through-fair unwind was losing edge. Replace with:
    1. QTY-based position control: reduce build-side qty as position grows
       (preserves full penny-jump edge on BOTH sides)
    2. Mild price tightening on unwind side (to fair+1, NEVER through fair)
    3. Position-aware taker: only take at fair in position-reducing direction
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

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        fair_floor = math.floor(fair)
        fair_ceil = math.ceil(fair)
        if fair == fair_floor:
            fair_ceil = fair_floor + 1

        # ========== 1) AGGRESSIVE TAKE ==========
        # Take anything strictly better than fair
        # Position-reducing: also take AT fair
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

        # Never cross fair
        pj_bid = min(pj_bid, fair_floor - 1)
        pj_ask = max(pj_ask, fair_ceil + 1)

        abs_pos = abs(pos)

        # --- PRICE: mild tightening on unwind side only ---
        # Only at high position, move unwind price toward fair+1
        # NEVER through fair
        if abs_pos > 25 and pos > 0:
            t = min((abs_pos - 25) / 25.0, 1.0)
            our_ask = round(pj_ask * (1 - t) + (fair_ceil + 1) * t)
            our_ask = max(our_ask, fair_ceil + 1)  # NEVER below fair+1
            our_bid = pj_bid
        elif abs_pos > 25 and pos < 0:
            t = min((abs_pos - 25) / 25.0, 1.0)
            our_bid = round(pj_bid * (1 - t) + (fair_floor - 1) * t)
            our_bid = min(our_bid, fair_floor - 1)  # NEVER above fair-1
            our_ask = pj_ask
        else:
            our_bid = pj_bid
            our_ask = pj_ask

        # Safety
        if our_bid >= our_ask:
            our_bid = fair_floor - 1
            our_ask = fair_ceil + 1

        # --- QTY: reduce build-side quantity based on position ---
        # Unwind side always gets full remaining capacity
        # Build side gets reduced as position grows
        if pos > 0:
            # Long: sell side = full, buy side = reduced
            sell_qty = limit + pos
            # Scale buy qty: full at pos=0, half at pos=25, zero at pos=50
            buy_frac = max(0.0, 1.0 - abs_pos / limit)
            buy_qty = max(0, min(round(buy_frac * (limit - pos)), limit - pos))
        elif pos < 0:
            # Short: buy side = full, sell side = reduced
            buy_qty = limit - pos
            sell_frac = max(0.0, 1.0 - abs_pos / limit)
            sell_qty = max(0, min(round(sell_frac * (limit + pos)), limit + pos))
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
