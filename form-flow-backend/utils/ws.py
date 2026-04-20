import asyncio
from collections import defaultdict
from typing import Dict, Set, Optional, Any

from fastapi import WebSocket

class ConnectionManager:
    """
    Lightweight WebSocket connection manager keyed by client_id.
    Allows broadcast to all or to a specific client.
    """

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str = "global"):
        """Accept connection and register client."""
        await websocket.accept()
        async with self._lock:
            self._connections[client_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, client_id: str = "global"):
        """Remove websocket from registry."""
        async with self._lock:
            if client_id in self._connections:
                self._connections[client_id].discard(websocket)
                if not self._connections[client_id]:
                    del self._connections[client_id]

    async def broadcast_json(self, message: Any, client_id: Optional[str] = None):
        """
        Broadcast a JSON-serializable message.
        If client_id is provided, only send to that client_id.
        """
        targets = []
        async with self._lock:
            if client_id:
                targets = list(self._connections.get(client_id, []))
            else:
                # Flatten all connections
                for conns in self._connections.values():
                    targets.extend(list(conns))

        # Collect dead connections to disconnect after iteration
        dead_connections = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                # Mark dead connection for removal after iteration
                dead_connections.append(ws)

        # Disconnect dead connections outside the broadcast loop
        for ws in dead_connections:
            await self.disconnect(ws, client_id or "global")


# Singleton manager
ws_manager = ConnectionManager()


async def emit_field_filled(field_id: str, value: Any, client_id: Optional[str] = None):
    """Helper to emit standardized field_filled messages."""
    await ws_manager.broadcast_json(
        {"type": "field_filled", "field_id": field_id, "value": value},
        client_id=client_id or "global",
    )
