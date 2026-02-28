from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
from utils.logging import get_logger

logger = get_logger(__name__)

class ConnectionManager:
    def __init__(self):
        # Maps user_id -> list of active websockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str = "anonymous"):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected: user={user_id}, total={len(self.active_connections[user_id])}")

    def disconnect(self, websocket: WebSocket, user_id: str = "anonymous"):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"WebSocket disconnected: user={user_id}")

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            msg_text = json.dumps(message)
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(msg_text)
                except Exception as e:
                    logger.error(f"Failed to send WS message: {e}")

manager = ConnectionManager()
router = APIRouter()

@router.websocket("/ws")
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str = "anonymous"):
    """
    WebSocket endpoint for real-time communication.
    Supports anonymous or authenticated connections.
    """
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message from {user_id}: {data}")
            
            # Message handling logic can be added here
            await websocket.send_text(f"Processed: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {e}")
        manager.disconnect(websocket, user_id)
