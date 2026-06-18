import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    Penny-jumping market maker v6 — aggressive position management.

    Core: penny-jump the bot to intercept aggressive flow.
    
    Key v6 change: GRADUATED AGGRESSIVE UNWIND.
    At high position, the opportunity cost of being at the limit is
    much higher than the cost of unwinding at a worse price.
    
    Position 0-20:  pure penny-jump both sides (max edge)
    Position 20-35: unwind side tightens toward fair (recover capacity)
    Position 35-50: unwind side AT or THROUGH fair (dump fast)
    
    Also: position-aware taker + fast TOMATO EMA.
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
        # Always take anything better than fair
        # Position-aware: when reducing position, also take AT fair
        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                should_buy = ask_px < fair or (ask_px <= fair and pos < 0)
                if should_buy:
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                should_sell = bid_px > fair or (bid_px >= fair and pos > 0)
                if should_sell:
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break

        # ========== 2) QUOTE WITH POSITION-PROPORTIONAL UNWIND ==========
        abs_pos = abs(pos)

        # --- Penny-jump base prices ---
        pj_bid = (best_bid + 1) if best_bid is not None else (fair_floor - 1)
        pj_ask = (best_ask - 1) if best_ask is not None else (fair_ceil + 1)
        pj_bid = min(pj_bid, fair_floor - 1)
        pj_ask = max(pj_ask, fair_ceil + 1)

        if abs_pos <= 20:
            # Low position: pure penny-jump both sides
            our_bid = pj_bid
            our_ask = pj_ask
        elif abs_pos <= 35:
            # Medium position: tighten unwind side toward fair
            t = (abs_pos - 20) / 15.0  # 0 → 1
            if pos > 0:
                # Long: tighten ask toward fair+1
                our_ask = round(pj_ask * (1 - t) + (fair_ceil + 1) * t)
                our_ask = max(our_ask, fair_ceil + 1)
                our_bid = pj_bid
            else:
                # Short: tighten bid toward fair-1
                our_bid = round(pj_bid * (1 - t) + (fair_floor - 1) * t)
                our_bid = min(our_bid, fair_floor - 1)
                our_ask = pj_ask
        else:
            # High position: unwind side at or through fair
            t = (abs_pos - 35) / 15.0  # 0 → 1
            if pos > 0:
                # Long: ask at fair, and as position grows, even below fair
                aggression = math.floor(t * 2)  # 0, 1, or 2 ticks through fair
                our_ask = fair_ceil - aggression
                our_ask = max(our_ask, fair_floor)  # don't go crazy
                our_bid = pj_bid
            else:
                # Short: bid at fair, pushing above fair as position grows
                aggression = math.floor(t * 2)
                our_bid = fair_floor + aggression
                our_bid = min(our_bid, fair_ceil)
                our_ask = pj_ask

        # Safety: never cross
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
