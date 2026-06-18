import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    Optimised market-making strategy for EMERALDS and TOMATOES.

    Key changes vs v1:
      - EMERALDS edge tightened 3->2 (quote 9998/10002 inside 9992/10008 bot spread)
      - EMERALDS quote sizes increased (20+15 instead of 15+10)
      - TOMATOES EMA alpha raised 0.15->0.35 for faster drift tracking
      - TOMATOES edge tightened to 2 primary / 4 secondary
      - Aggressive taker sweeps orders AT fair value (not just beyond)
      - Stronger position-skew so we don't get stuck directional
    """

    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}

    # -- EMERALDS --
    EMERALD_FAIR = 10000
    EMERALD_EDGE = 2          # primary: 9998 / 10002
    EMERALD_EDGE_WIDE = 4     # secondary: 9996 / 10004

    # -- TOMATOES --
    TOMATO_EMA_ALPHA = 0.35   # fast-tracking EMA
    TOMATO_EDGE = 2
    TOMATO_EDGE_WIDE = 4

    # -- position management --
    SKEW_PER_UNIT = 0.06      # price shift per unit of position

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

    # -----------------------------------------
    #  EMERALDS  -  fixed fair = 10 000
    # -----------------------------------------
    def _trade_emeralds(self, state: TradingState, data: dict) -> list[Order]:
        product = "EMERALDS"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        fair = self.EMERALD_FAIR
        orders: list[Order] = []

        # 1) Aggressive take -- sweep everything at or better than fair
        orders += self._take_at_or_better(od, fair, product, pos, limit)
        pos = self._proj(pos, orders)

        # 2) Position-dependent skew
        skew = -self.SKEW_PER_UNIT * pos

        # 3) Primary quotes (tight)
        bp1 = math.floor(fair - self.EMERALD_EDGE + skew)
        sp1 = math.ceil(fair + self.EMERALD_EDGE + skew)
        bq1 = min(20, limit - pos)
        sq1 = min(20, limit + pos)
        if bq1 > 0:
            orders.append(Order(product, bp1, bq1))
        if sq1 > 0:
            orders.append(Order(product, sp1, -sq1))

        pos = self._proj(state.position.get(product, 0), orders)

        # 4) Secondary quotes (wider, fill remaining capacity)
        bp2 = math.floor(fair - self.EMERALD_EDGE_WIDE + skew)
        sp2 = math.ceil(fair + self.EMERALD_EDGE_WIDE + skew)
        bq2 = min(15, limit - pos)
        sq2 = min(15, limit + pos)
        if bq2 > 0 and bp2 < bp1:
            orders.append(Order(product, bp2, bq2))
        if sq2 > 0 and sp2 > sp1:
            orders.append(Order(product, sp2, -sq2))

        return orders

    # -----------------------------------------
    #  TOMATOES  -  EMA-tracked fair value
    # -----------------------------------------
    def _trade_tomatoes(self, state: TradingState, data: dict) -> list[Order]:
        product = "TOMATOES"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []

        mid = self._mid(od)
        if mid is None:
            return orders

        # EMA update
        k = "t_ema"
        ema = data.get(k, mid)
        ema = self.TOMATO_EMA_ALPHA * mid + (1 - self.TOMATO_EMA_ALPHA) * ema
        data[k] = ema
        fair = ema

        # 1) Aggressive take at or better than fair
        orders += self._take_at_or_better(od, fair, product, pos, limit)
        pos = self._proj(pos, orders)

        # 2) Position-dependent skew
        skew = -self.SKEW_PER_UNIT * pos

        # 3) Primary quotes
        bp1 = math.floor(fair - self.TOMATO_EDGE + skew)
        sp1 = math.ceil(fair + self.TOMATO_EDGE + skew)
        bq1 = min(15, limit - pos)
        sq1 = min(15, limit + pos)
        if bq1 > 0:
            orders.append(Order(product, bp1, bq1))
        if sq1 > 0:
            orders.append(Order(product, sp1, -sq1))

        pos = self._proj(state.position.get(product, 0), orders)

        # 4) Secondary quotes
        bp2 = math.floor(fair - self.TOMATO_EDGE_WIDE + skew)
        sp2 = math.ceil(fair + self.TOMATO_EDGE_WIDE + skew)
        bq2 = min(10, limit - pos)
        sq2 = min(10, limit + pos)
        if bq2 > 0 and bp2 < bp1:
            orders.append(Order(product, bp2, bq2))
        if sq2 > 0 and sp2 > sp1:
            orders.append(Order(product, sp2, -sq2))

        return orders

    # -----------------------------------------
    #  Helpers
    # -----------------------------------------
    @staticmethod
    def _mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def _proj(pos: int, orders: list[Order]) -> int:
        return pos + sum(o.quantity for o in orders)

    @staticmethod
    def _take_at_or_better(od: OrderDepth, fair: float, product: str,
                           pos: int, limit: int) -> list[Order]:
        """
        Sweep ALL resting orders priced at fair or better.
        Buy asks <= fair, sell into bids >= fair.
        """
        orders: list[Order] = []

        # Buy cheap asks  (ask_price <= fair)
        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px <= fair:
                    vol = -od.sell_orders[ask_px]
                    can = limit - pos
                    qty = min(vol, can)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        # Sell into rich bids  (bid_price >= fair)
        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px >= fair:
                    vol = od.buy_orders[bid_px]
                    can = limit + pos
                    qty = min(vol, can)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break

        return orders
