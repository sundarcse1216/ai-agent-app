import sqlite3

from config import EVENTS_DATABASE_PATH


class EventRepository:

    def _connect(self):
        return sqlite3.connect(EVENTS_DATABASE_PATH)

    def _fetch(self, query, params=()):
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as e:
            raise Exception(f"Database error: {e}")

    def get_all_events(self):
        return self._fetch("SELECT * FROM events")

    def get_events(self, date):
        return self._fetch(
            "SELECT * FROM events WHERE date = ?",
            (date,)
        )

    def get_events_by_type(self, date, event_type):
        return self._fetch(
            "SELECT * FROM events WHERE date = ? AND type = ?",
            (date, event_type)
        )
