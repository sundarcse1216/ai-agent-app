import sqlite3

from config import DATABASE_PATH


def setup_company_database():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Create sample tables 
    c.execute('''
              CREATE TABLE IF NOT EXISTS employees
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY,
                  name
                  TEXT,
                  department
                  TEXT,
                  salary
                  REAL
              )
              ''')

    c.execute('''
              CREATE TABLE IF NOT EXISTS departments
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY,
                  name
                  TEXT,
                  budget
                  REAL
              )
              ''')

    # Insert sample data 
    c.execute("INSERT OR IGNORE INTO employees VALUES (1, 'John Doe', 'Engineering', 75000)")
    c.execute("INSERT OR IGNORE INTO employees VALUES (2, 'Jane Smith', 'Marketing', 65000)")
    c.execute("INSERT OR IGNORE INTO departments VALUES (1, 'Engineering', 1000000)")
    c.execute("INSERT OR IGNORE INTO departments VALUES (2, 'Marketing', 500000)")

    conn.commit()
    conn.close()
