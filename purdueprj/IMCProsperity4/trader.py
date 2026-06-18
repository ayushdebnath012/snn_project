import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order


class Trader:
    """
    Trading strategy for EMERALDS and TOMATOES.

    EMERALDS — stationary fair value at 10000, tight mean reversion.
               Market-make inside the wide bot spread (9992/10008).

    TOMATOES — drifting fair value tracked with an EMA.
               Market-make around the EMA, skew quotes with position.
    """

    # ── position limits (per product) ──
    LIMITS = {
        "EMERALDS": 50,
        "TOMATOES": 50,
    }

    # ── EMERALDS params ──
    EMERALD_FAIR = 10000
    EMERALD_EDGE = 3          # quote this far from fair (buy 9997, sell 10003)
    EMERALD_EDGE_WIDE = 6     # secondary layer

    # ── TOMATOES params ──
    TOMATO_EMA_ALPHA = 0.15   # responsive EMA
    TOMATO_EDGE = 3
    TOMATO_EDGE_WIDE = 5

    # ── position-skew intensity ──
    SKEW_INTENSITY = 0.4      # how much to shift quotes per unit of position / limit

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        # ── restore persistent state ──
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        # ── EMERALDS ──
        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self._trade_emeralds(state, data)

        # ── TOMATOES ──
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self._trade_tomatoes(state, data)

        trader_data = json.dumps(data)
        return result, conversions, trader_data

    # ──────────────────────────────────────────────
    #  EMERALDS  –  fixed fair-value market making
    # ──────────────────────────────────────────────
    def _trade_emeralds(self, state: TradingState, data: dict) -> list[Order]:
        product = "EMERALDS"
        orders: list[Order] = []
        order_depth: OrderDepth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]
        fair = self.EMERALD_FAIR

        # ---- aggressive take: hit any mispriced resting orders ----
        orders += self._take_mispriced(order_depth, fair, product, position, limit)
        position = self._projected_position(position, orders)

        # ---- passive market-making layer ----
        skew = self._position_skew(position, limit) * 2  # stronger skew for emeralds

        buy_price  = round(fair - self.EMERALD_EDGE + skew)
        sell_price = round(fair + self.EMERALD_EDGE + skew)

        buy_qty  = min(15, limit - position)
        sell_qty = min(15, limit + position)

        if buy_qty > 0:
            orders.append(Order(product, buy_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, sell_price, -sell_qty))

        # ---- wider secondary layer ----
        position2 = self._projected_position(state.position.get(product, 0), orders)
        buy_price2  = round(fair - self.EMERALD_EDGE_WIDE + skew)
        sell_price2 = round(fair + self.EMERALD_EDGE_WIDE + skew)

        buy_qty2  = min(10, limit - position2)
        sell_qty2 = min(10, limit + position2)

        if buy_qty2 > 0 and buy_price2 < buy_price:
            orders.append(Order(product, buy_price2, buy_qty2))
        if sell_qty2 > 0 and sell_price2 > sell_price:
            orders.append(Order(product, sell_price2, -sell_qty2))

        return orders

    # ──────────────────────────────────────────────
    #  TOMATOES  –  EMA-tracking market making
    # ──────────────────────────────────────────────
    def _trade_tomatoes(self, state: TradingState, data: dict) -> list[Order]:
        product = "TOMATOES"
        orders: list[Order] = []
        order_depth: OrderDepth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]

        # compute mid price from order book
        mid = self._mid_price(order_depth)
        if mid is None:
            return orders

        # update EMA
        ema_key = "tomato_ema"
        if ema_key in data:
            ema = data[ema_key] * (1 - self.TOMATO_EMA_ALPHA) + mid * self.TOMATO_EMA_ALPHA
        else:
            ema = mid
        data[ema_key] = ema

        fair = ema

        # ---- aggressive take: hit mispriced resting orders ----
        orders += self._take_mispriced(order_depth, fair, product, position, limit)
        position = self._projected_position(position, orders)

        # ---- passive market-making ----
        skew = self._position_skew(position, limit) * 1.5

        buy_price  = round(fair - self.TOMATO_EDGE + skew)
        sell_price = round(fair + self.TOMATO_EDGE + skew)

        buy_qty  = min(12, limit - position)
        sell_qty = min(12, limit + position)

        if buy_qty > 0:
            orders.append(Order(product, buy_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, sell_price, -sell_qty))

        # ---- wider layer ----
        position2 = self._projected_position(state.position.get(product, 0), orders)
        buy_price2  = round(fair - self.TOMATO_EDGE_WIDE + skew)
        sell_price2 = round(fair + self.TOMATO_EDGE_WIDE + skew)

        buy_qty2  = min(8, limit - position2)
        sell_qty2 = min(8, limit + position2)

        if buy_qty2 > 0 and buy_price2 < buy_price:
            orders.append(Order(product, buy_price2, buy_qty2))
        if sell_qty2 > 0 and sell_price2 > sell_price:
            orders.append(Order(product, sell_price2, -sell_qty2))

        return orders

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────
    @staticmethod
    def _mid_price(od: OrderDepth) -> float | None:
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            return (best_bid + best_ask) / 2.0
        return None

    def _position_skew(self, position: int, limit: int) -> float:
        """Negative when long (push quotes down to sell), positive when short."""
        return -self.SKEW_INTENSITY * (position / limit)

    @staticmethod
    def _projected_position(current_pos: int, orders: list[Order]) -> int:
        """Estimate position after all orders fill."""
        return current_pos + sum(o.quantity for o in orders)

    @staticmethod
    def _take_mispriced(od: OrderDepth, fair: float, product: str,
                        position: int, limit: int) -> list[Order]:
        """
        Aggressively take any resting orders that are mispriced vs. fair value.
        Buy anything offered below fair, sell anything bid above fair.
        """
        orders: list[Order] = []

        # Buy cheap asks (ask < fair)
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair:
                    ask_vol = -od.sell_orders[ask_price]  # sell_orders have negative qty
                    can_buy = limit - position
                    qty = min(ask_vol, can_buy)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        position += qty

        # Sell into rich bids (bid > fair)
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair:
                    bid_vol = od.buy_orders[bid_price]
                    can_sell = limit + position  # position can go to -limit
                    qty = min(bid_vol, can_sell)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        position -= qty

        return orders
