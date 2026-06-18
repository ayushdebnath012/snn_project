import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    v7 — Fixed the self-penny-jump bug.

    Instead of penny-jumping the best bid/ask (which includes our own
    resting orders), we now quote at FIXED offsets from fair value,
    calibrated to sit just inside the bot's spread.

    EMERALDS: bot at 9992/10008 → we quote 9998/10002 (edge 4 each side)
              Actually no — we want max edge. Bot is at 9992/10008.
              We quote 9993/10007 ALWAYS. Hardcoded. No penny-jump.

    TOMATOES: bot at mid±6.5 to mid±7 → we quote fair±6 ALWAYS.

    Position management: graduated unwind unchanged from v6.
    """

    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}
    EMERALD_FAIR = 10000
    TOMATO_EMA_ALPHA = 0.50

    # Fixed offsets from fair (just inside the bot spread)
    EMERALD_OFFSET = 7   # quote at 9993/10007
    TOMATO_OFFSET = 6    # quote at fair-6/fair+6

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
                state, "EMERALDS", self.EMERALD_FAIR,
                self.EMERALD_OFFSET, data
            )
        if "TOMATOES" in state.order_depths:
            od = state.order_depths["TOMATOES"]
            mid = self._mid(od)
            if mid is not None:
                k = "t_ema"
                ema = data.get(k, mid)
                ema = self.TOMATO_EMA_ALPHA * mid + (1 - self.TOMATO_EMA_ALPHA) * ema
                data[k] = ema
                result["TOMATOES"] = self._trade(
                    state, "TOMATOES", ema,
                    self.TOMATO_OFFSET, data
                )
            else:
                result["TOMATOES"] = []

        return result, conversions, json.dumps(data)

    def _trade(self, state: TradingState, product: str, fair: float,
               offset: int, data: dict) -> list[Order]:
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []

        # ========== 1) AGGRESSIVE TAKE ==========
        # Buy anything below fair (position-reducing: also at fair)
        # Sell anything above fair (position-reducing: also at fair)
        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                take = ask_px < fair or (ask_px <= fair and pos < 0)
                if take:
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                take = bid_px > fair or (bid_px >= fair and pos > 0)
                if take:
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break

        # ========== 2) FIXED-OFFSET QUOTING WITH GRADUATED UNWIND ==========
        abs_pos = abs(pos)

        # Base quote prices: fixed offset from fair (NO penny-jumping)
        base_bid = math.floor(fair) - offset
        base_ask = math.ceil(fair) + offset
        if math.ceil(fair) == math.floor(fair):
            base_ask = int(fair) + offset

        if abs_pos <= 15:
            # Comfortable: full edge both sides
            our_bid = base_bid
            our_ask = base_ask
        elif abs_pos <= 30:
            # Medium: tighten unwind side toward fair
            t = (abs_pos - 15) / 15.0
            if pos > 0:
                # Long → tighten ask
                tight_ask = math.ceil(fair) + 1
                our_ask = round(base_ask * (1 - t) + tight_ask * t)
                our_ask = max(our_ask, math.ceil(fair) + 1)
                our_bid = base_bid
            else:
                # Short → tighten bid
                tight_bid = math.floor(fair) - 1
                our_bid = round(base_bid * (1 - t) + tight_bid * t)
                our_bid = min(our_bid, math.floor(fair) - 1)
                our_ask = base_ask
        else:
            # High position: unwind side at/through fair
            t = min((abs_pos - 30) / 20.0, 1.0)
            aggression = math.floor(t * 2)
            if pos > 0:
                our_ask = math.ceil(fair) + 1 - aggression
                our_ask = max(our_ask, math.floor(fair))
                our_bid = base_bid
            else:
                our_bid = math.floor(fair) - 1 + aggression
                our_bid = min(our_bid, math.ceil(fair))
                our_ask = base_ask

        # Safety: never cross
        if our_bid >= our_ask:
            our_bid = math.floor(fair) - 1
            our_ask = math.ceil(fair) + 1

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
