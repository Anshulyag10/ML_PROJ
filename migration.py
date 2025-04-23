# migration.py
import sqlite3
import os

def migrate_database():
    """Add missing columns to the user table"""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('cosmic_finance.db')
        cursor = conn.cursor()
        
        print("Connected to database")
        
        # Check if User table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cursor.fetchone():
            print("Creating User table")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user (
                    id INTEGER PRIMARY KEY,
                    username VARCHAR(64) UNIQUE,
                    email VARCHAR(120) UNIQUE,
                    password_hash VARCHAR(128),
                    created_at TIMESTAMP
                )
            ''')
        
        # Check if columns exist
        cursor.execute("PRAGMA table_info(user)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Existing columns: {columns}")
        
        # Add missing columns if needed
        if 'reset_token' not in columns:
            print("Adding reset_token column")
            cursor.execute("ALTER TABLE user ADD COLUMN reset_token VARCHAR")
        
        if 'reset_token_expiry' not in columns:
            print("Adding reset_token_expiry column")
            cursor.execute("ALTER TABLE user ADD COLUMN reset_token_expiry TIMESTAMP")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        print("Database migration completed successfully")
        
    except Exception as e:
        print(f"Error during database migration: {e}")

if __name__ == "__main__":
    migrate_database()
