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

    # Sample events
    events = [
        ("Summer Concert", "outdoor", "Live music in the park", "Central Park", today.isoformat(), 25.00),
        ("Art Exhibition", "indoor", "Modern art showcase", "City Gallery", (today + timedelta(days=1)).isoformat(), 50.00),
        ("Food Festival", "outdoor", "International cuisine", "Waterfront", (today + timedelta(days=2)).isoformat(), 10.00),
        ("Theater Show", "indoor", "Classical drama", "Grand Theater", (today + timedelta(days=3)).isoformat(), 15.00),
    ]

    c.executemany('INSERT OR IGNORE INTO events (name, type, description, location, date, price) VALUES (?,?,?,?,?,?)',
                  events)
    conn.commit()
    conn.close()
