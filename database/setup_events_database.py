import sqlite3
from datetime import date, timedelta

from config import EVENTS_DATABASE_PATH

today = date.today()


def setup_events_database():
    conn = sqlite3.connect(EVENTS_DATABASE_PATH)
    c = conn.cursor()

    c.execute('''
              CREATE TABLE IF NOT EXISTS events
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY,
                  name
                  TEXT,
                  type
                  TEXT, -- 'indoor' or 'outdoor' 
                  description
                  TEXT,
                  location
                  TEXT,
                  date
                  TEXT,
                  price
                  REAL
              )
              ''')

    # Sample events, pinned to fixed ids 1-4 so re-running this (idempotent,
    # called on every app startup) refreshes their dates to stay relative to
    # "today" via OR REPLACE, instead of silently accumulating a new
    # duplicate batch of rows every day (which OR IGNORE without explicit
    # ids would do, since new autoincrement rowids never collide).
    events = [
        (1, "Summer Concert", "outdoor", "Live music in the park", "Central Park", today.isoformat(), 25.00),
        (2, "Art Exhibition", "indoor", "Modern art showcase", "City Gallery", (today + timedelta(days=1)).isoformat(), 50.00),
        (3, "Food Festival", "outdoor", "International cuisine", "Waterfront", (today + timedelta(days=2)).isoformat(), 10.00),
        (4, "Theater Show", "indoor", "Classical drama", "Grand Theater", (today + timedelta(days=3)).isoformat(), 15.00),
    ]

    c.executemany('INSERT OR REPLACE INTO events (id, name, type, description, location, date, price) VALUES (?,?,?,?,?,?,?)',
                  events)
    conn.commit()
    conn.close()
