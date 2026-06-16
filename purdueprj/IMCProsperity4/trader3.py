import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    Penny-jumping market maker.

    EMERALDS — fair = 10000, bot quotes 9992/10008.
               We quote 9993/10007 → profit 14 per round trip.
    TOMATOES — fair = EMA, bot quotes ~mid-7/mid+7.
               We quote 1 tick inside → profit ~12 per round trip.

    Adaptive: reads actual order book and undercuts by 1 tick.
    Position skew prevents directional blowup.
    """

    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}

    EMERALD_FAIR = 10000
    TOMATO_EMA_ALPHA = 0.35

    # Skew: shift mid by this much per unit of position
    SKEW_PER_UNIT = 0.15

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
            result["EMERALDS"] = self._trade_emeralds(state, data)
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self._trade_tomatoes(state, data)

        return result, conversions, json.dumps(data)

    def _trade_emeralds(self, state: TradingState, data: dict) -> list[Order]:
        product = "EMERALDS"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        fair = self.EMERALD_FAIR
        orders: list[Order] = []

        # 1) Aggressive take — buy any ask < fair, sell any bid > fair
        orders += self._take_mispriced(od, fair, product, pos, limit)
        pos = self._proj(pos, orders)

        # 2) Penny-jump: undercut the best resting bid/ask by 1
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        # Position skew shifts our effective fair value
        skew = -self.SKEW_PER_UNIT * pos

        # Our bid = 1 better than current best bid, but never above fair+skew
        if best_bid is not None:
            our_bid = min(best_bid + 1, math.floor(fair + skew) - 1)
        else:
            our_bid = math.floor(fair - 1 + skew)

        # Our ask = 1 better than current best ask, but never below fair+skew
        if best_ask is not None:
            our_ask = max(best_ask - 1, math.ceil(fair + skew) + 1)
        else:
            our_ask = math.ceil(fair + 1 + skew)

        # Ensure we don't cross ourselves
        if our_bid >= our_ask:
            our_bid = math.floor(fair + skew) - 1
            our_ask = math.ceil(fair + skew) + 1

        buy_qty = limit - pos
        sell_qty = limit + pos

        if buy_qty > 0:
            orders.append(Order(product, our_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, our_ask, -sell_qty))

        return orders

    def _trade_tomatoes(self, state: TradingState, data: dict) -> list[Order]:
        product = "TOMATOES"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []

        mid = self._mid(od)
        if mid is None:
            return orders

        # EMA for fair value
        k = "t_ema"
        ema = data.get(k, mid)
        ema = self.TOMATO_EMA_ALPHA * mid + (1 - self.TOMATO_EMA_ALPHA) * ema
        data[k] = ema
        fair = ema

        # 1) Aggressive take
        orders += self._take_mispriced(od, fair, product, pos, limit)
        pos = self._proj(pos, orders)

        # 2) Penny-jump
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        skew = -self.SKEW_PER_UNIT * pos

        if best_bid is not None:
            our_bid = min(best_bid + 1, math.floor(fair + skew) - 1)
        else:
            our_bid = math.floor(fair - 1 + skew)

        if best_ask is not None:
            our_ask = max(best_ask - 1, math.ceil(fair + skew) + 1)
        else:
            our_ask = math.ceil(fair + 1 + skew)

        if our_bid >= our_ask:
            our_bid = math.floor(fair + skew) - 1
            our_ask = math.ceil(fair + skew) + 1

        buy_qty = limit - pos
        sell_qty = limit + pos

        if buy_qty > 0:
            orders.append(Order(product, our_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, our_ask, -sell_qty))

        return orders

    # --- Helpers ---
    @staticmethod
    def _mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def _proj(pos: int, orders: list[Order]) -> int:
        return pos + sum(o.quantity for o in orders)

    @staticmethod
    def _take_mispriced(od: OrderDepth, fair: float, product: str,
                        pos: int, limit: int) -> list[Order]:
        orders: list[Order] = []
        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px < fair:
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break
        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px > fair:
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break
        return orders
