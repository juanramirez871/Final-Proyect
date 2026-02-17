from threading import Lock
from typing import Dict, List


class ConversationStore:
    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._lock = Lock()

    def add_user_message(self, session_id: str, text: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append({"role": "user", "text": text})

    def add_model_message(self, session_id: str, text: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append({"role": "assistant", "text": text})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]

    def clear_all(self) -> None:
        with self._lock:
            self._store.clear()
