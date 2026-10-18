"""Listen to League Client Update (LCU) champion-select events.

The listener reports champion IDs even when name lookup is unavailable. Names
are resolved from Riot's Data Dragon static champion list on first use and
cached for the lifetime of the process.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from lcu_driver import Connector

logger = logging.getLogger(__name__)

Matchup = dict[str, Any]
MatchupCallback = Callable[[Matchup], None | Awaitable[None]]


class LCUClientListener:
    """Bridge LCU champ-select and game-flow updates to an application callback.

    The callback receives the local champion, visible opposing champions, and
    game-flow phase. ``enemy_champion`` is the best single-opponent guess when
    a lane match can be inferred; ``enemy_champions`` always contains all
    revealed enemies. All champion IDs are included as a safe fallback.

    Args:
        on_matchup: Called whenever a useful matchup or game-flow update arrives.
            It may be a normal function or an async function.
        resolve_names: Whether to resolve champion IDs to names from the
            Data Dragon static champion list. If unavailable, IDs are kept.
    """

    CHAMP_SELECT_URI = "/lol-champ-select/v1/session"
    GAMEFLOW_PHASE_URI = "/lol-gameflow/v1/gameflow-phase"
    GAMEFLOW_SESSION_URI = "/lol-gameflow/v1/session"
    DATADRAGON_VERSIONS_URI = "https://ddragon.leagueoflegends.com/api/versions.json"
    DATADRAGON_CHAMPIONS_URI = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"

    def __init__(
        self,
        on_matchup: MatchupCallback,
        *,
        resolve_names: bool = True,
    ) -> None:
        self.on_matchup = on_matchup
        self.resolve_names = resolve_names
        self.connector = Connector(loop=asyncio.new_event_loop())
        self._connection: Any | None = None
        self._summoner_id: int | None = None
        self._champion_names: dict[int, str] = {}
        self._last_matchup: Matchup = {
            "player_champion": None,
            "player_champion_id": None,
            "enemy_champion": None,
            "enemy_champion_id": None,
            "enemy_champions": [],
            "game_phase": None,
        }
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.connector.ready
        async def on_ready(connection: Any) -> None:
            self._connection = connection
            logger.info("Connected to the League Client API.")
            await self._load_summoner_id(connection)
            await self._load_champion_names()
            await self._refresh_champ_select(connection)
            await self._refresh_game_phase(connection)

        @self.connector.close
        async def on_close(_: Any) -> None:
            self._connection = None
            logger.info("Disconnected from the League Client API; waiting for the client.")

        @self.connector.ws.register(
            self.CHAMP_SELECT_URI,
            event_types=("CREATE", "UPDATE", "DELETE"),
        )
        async def on_champ_select(_: Any, event: Any) -> None:
            data = getattr(event, "data", None)
            if isinstance(data, dict):
                await self._publish_champ_select(data)
            elif getattr(event, "event_type", None) == "DELETE":
                self._last_matchup = {
                    "player_champion": None,
                    "player_champion_id": None,
                    "enemy_champion": None,
                    "enemy_champion_id": None,
                    "enemy_champions": [],
                    "game_phase": self._last_matchup.get("game_phase"),
                }

        @self.connector.ws.register(self.GAMEFLOW_PHASE_URI, event_types=("UPDATE",))
        async def on_gameflow_phase(_: Any, event: Any) -> None:
            phase = getattr(event, "data", None)
            if isinstance(phase, str):
                await self._publish_phase(phase)

        @self.connector.ws.register(self.GAMEFLOW_SESSION_URI, event_types=("CREATE", "UPDATE"))
        async def on_gameflow_session(_: Any, event: Any) -> None:
            data = getattr(event, "data", None)
            if isinstance(data, dict):
                phase = data.get("phase") or data.get("gameData", {}).get("phase")
                if isinstance(phase, str):
                    await self._publish_phase(phase)

    def start(self) -> None:
        """Start lcu-driver's reconnecting client listener.

        ``lcu-driver`` owns and runs its asyncio event loop inside this call,
        so applications with a UI should invoke ``start`` on a background
        thread.
        """
        logger.info("Waiting for the League Client to become available...")
        self.connector.start()

    def stop(self) -> None:
        """Stop the connector, including when called from a UI thread."""
        loop = self.connector.loop
        if loop.is_closed():
            return
        coroutine = self.connector.stop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coroutine, loop)
        else:
            loop.run_until_complete(coroutine)

    async def _load_summoner_id(self, connection: Any) -> None:
        try:
            response = await connection.request("get", "/lol-summoner/v1/current-summoner")
            if response.status == 200:
                summoner = await response.json()
                value = summoner.get("summonerId")
                self._summoner_id = int(value) if value is not None else None
        except Exception:
            logger.exception("Could not read the current summoner from the LCU.")

    async def _load_champion_names(self) -> None:
        if not self.resolve_names or self._champion_names:
            return
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.DATADRAGON_VERSIONS_URI) as response:
                    response.raise_for_status()
                    versions = await response.json()
                if not isinstance(versions, list) or not versions:
                    raise ValueError("Data Dragon did not return a version list")
                champion_uri = self.DATADRAGON_CHAMPIONS_URI.format(version=versions[0])
                async with session.get(champion_uri) as response:
                    response.raise_for_status()
                    payload = await response.json()
            champions = payload.get("data", {})
            self._champion_names = {
                int(item["key"]): item["name"]
                for item in champions.values()
                if isinstance(item, dict)
                and str(item.get("key", "")).isdigit()
                and item.get("name")
            }
            logger.info("Loaded names for %d champions.", len(self._champion_names))
        except Exception:
            # LCU event delivery should continue when static data is offline.
            logger.warning("Champion name lookup failed; reporting champion IDs instead.")

    async def _refresh_champ_select(self, connection: Any) -> None:
        try:
            response = await connection.request("get", self.CHAMP_SELECT_URI)
            if response.status == 200:
                data = await response.json()
                if isinstance(data, dict):
                    await self._publish_champ_select(data)
        except Exception:
            # The endpoint returns an error outside champion select, which is normal.
            logger.debug("No active champion-select session.", exc_info=True)

    async def _refresh_game_phase(self, connection: Any) -> None:
        try:
            response = await connection.request("get", self.GAMEFLOW_PHASE_URI)
            if response.status == 200:
                phase = await response.json()
                if isinstance(phase, str):
                    await self._publish_phase(phase)
        except Exception:
            logger.debug("Could not read the current game-flow phase.", exc_info=True)

    async def _publish_champ_select(self, session: dict[str, Any]) -> None:
        my_team = session.get("myTeam") or []
        their_team = session.get("theirTeam") or []
        local_player = self._find_local_player(my_team, session)

        player_id = self._selected_id(local_player) if local_player else None
        player_lane = self._lane(local_player) if local_player else None
        enemies = [entry for entry in their_team if self._selected_id(entry) is not None]
        enemies_data = [self._champion_record(entry) for entry in enemies]
        opponent = next(
            (entry for entry in enemies if player_lane and self._lane(entry) == player_lane),
            enemies[0] if len(enemies) == 1 else None,
        )
        enemy_id = self._selected_id(opponent) if opponent else None

        await self._emit(
            {
                "player_champion": self._champion_name(player_id),
                "player_champion_id": player_id,
                "player_locked_in": bool(local_player and local_player.get("championId")),
                "enemy_champion": self._champion_name(enemy_id),
                "enemy_champion_id": enemy_id,
                "enemy_champions": enemies_data,
                "player_lane": player_lane,
                "game_phase": self._last_matchup.get("game_phase"),
            }
        )

    async def _publish_phase(self, phase: str) -> None:
        self._last_matchup = {**self._last_matchup, "game_phase": phase}
        await self._emit(self._last_matchup)

    async def _emit(self, matchup: Matchup) -> None:
        self._last_matchup = matchup
        try:
            result = self.on_matchup(matchup.copy())
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("The matchup callback raised an error.")

    def _find_local_player(
        self,
        my_team: list[dict[str, Any]],
        session: dict[str, Any],
    ) -> dict[str, Any] | None:
        local_cell = session.get("localPlayerCellId")
        if local_cell is not None:
            for entry in my_team:
                if entry.get("cellId") == local_cell:
                    return entry
        if self._summoner_id is not None:
            for entry in my_team:
                if str(entry.get("summonerId")) == str(self._summoner_id):
                    return entry
        return None

    @staticmethod
    def _selected_id(entry: dict[str, Any] | None) -> int | None:
        if not entry:
            return None
        # Prefer the committed pick; otherwise expose the current hover selection.
        value = entry.get("championId") or entry.get("selectedChampionId")
        try:
            champion_id = int(value)
        except (TypeError, ValueError):
            return None
        return champion_id or None

    @staticmethod
    def _lane(entry: dict[str, Any] | None) -> str | None:
        if not entry:
            return None
        lane = entry.get("assignedPosition") or entry.get("position")
        return lane.lower() if isinstance(lane, str) and lane else None

    def _champion_name(self, champion_id: int | None) -> str | None:
        if champion_id is None:
            return None
        return self._champion_names.get(champion_id, f"Champion {champion_id}")

    def _champion_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        champion_id = self._selected_id(entry)
        return {
            "champion": self._champion_name(champion_id),
            "champion_id": champion_id,
            "lane": self._lane(entry),
        }


def _run_standalone() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def print_matchup(matchup: Matchup) -> None:
        print("\nMatchup update:")
        print(f"  Phase: {matchup.get('game_phase')}")
        print(f"  You: {matchup.get('player_champion') or 'not selected'}")
        print(f"  Best lane matchup: {matchup.get('enemy_champion') or 'not known yet'}")
        print(f"  Revealed enemy team: {matchup.get('enemy_champions')}")

    listener = LCUClientListener(print_matchup)
    try:
        listener.start()
    except KeyboardInterrupt:
        logger.info("Stopping LeagueBlueprint listener.")
        listener.stop()


if __name__ == "__main__":
    _run_standalone()
