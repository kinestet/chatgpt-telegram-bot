import logging
import psycopg2
from psycopg2 import pool
from datetime import date
import json

class DatabaseManager:
    """
    DatabaseManager class
    Handles all database operations for the usage tracker.
    """
    
    def __init__(self, config):
        """
        Initialize the database manager with connection parameters
        :param config: Dictionary containing database configuration
        """
        self.config = config
        self.connection_pool = None
        
        try:
            # Create a connection pool
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=config.get('db_host', 'localhost'),
                port=config.get('db_port', 5432),
                database=config.get('db_name', 'telegram_bot'),
                user=config.get('db_user', 'postgres'),
                password=config.get('db_password', '')
            )
            
            # Initialize database tables
            self._initialize_tables()
            logging.info("Database connection established successfully")
        except Exception as e:
            self.connection_pool = None
            logging.error(f"Failed to connect to database: {e}")
    
    def is_connected(self):
        """
        Check if the database connection is active
        :return: True if connected, False otherwise
        """
        return self.connection_pool is not None
    
    def _initialize_tables(self):
        """
        Initialize database tables if they don't exist
        """
        if not self.is_connected():
            return
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Create users table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    user_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_initiated_message TEXT
                )
                """)
                
                # Create token usage table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    usage_date DATE NOT NULL,
                    tokens INTEGER NOT NULL,
                    UNIQUE(user_id, usage_date)
                )
                """)
                
                # Create image usage table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_usage (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    usage_date DATE NOT NULL,
                    size_256 INTEGER DEFAULT 0,
                    size_512 INTEGER DEFAULT 0,
                    size_1024 INTEGER DEFAULT 0,
                    UNIQUE(user_id, usage_date)
                )
                """)
                
                # Create transcription usage table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcription_usage (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    usage_date DATE NOT NULL,
                    seconds FLOAT NOT NULL,
                    UNIQUE(user_id, usage_date)
                )
                """)
                
                # Create vision usage table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS vision_usage (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    usage_date DATE NOT NULL,
                    tokens INTEGER NOT NULL,
                    UNIQUE(user_id, usage_date)
                )
                """)
                
                # Create TTS usage table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS tts_usage (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    usage_date DATE NOT NULL,
                    model TEXT NOT NULL,
                    characters INTEGER NOT NULL,
                    UNIQUE(user_id, usage_date, model)
                )
                """)
                
                # Create cost tracking table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    day_cost FLOAT DEFAULT 0.0,
                    month_cost FLOAT DEFAULT 0.0,
                    all_time_cost FLOAT DEFAULT 0.0,
                    last_update DATE NOT NULL,
                    UNIQUE(user_id)
                )
                """)
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error initializing database tables: {e}")
        finally:
            self.connection_pool.putconn(conn)
    
    def get_or_create_user(self, user_id, user_name):
        """
        Get or create a user in the database
        :param user_id: Telegram user ID
        :param user_name: Telegram user name
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Check if user exists
                cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
                if cursor.fetchone() is None:
                    # Create new user
                    cursor.execute(
                        "INSERT INTO users (user_id, user_name) VALUES (%s, %s)",
                        (user_id, user_name)
                    )
                    
                    # Initialize cost tracking
                    today = date.today()
                    cursor.execute(
                        "INSERT INTO cost_tracking (user_id, last_update) VALUES (%s, %s)",
                        (user_id, today)
                    )
                else:
                    # Update user name if it changed
                    cursor.execute(
                        "UPDATE users SET user_name = %s WHERE user_id = %s",
                        (user_name, user_id)
                    )
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error creating/updating user: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def get_last_initiated_message(self, user_id):
        """
        Get the last initiated message date for a user
        :param user_id: Telegram user ID
        :return: Date string or None
        """
        if not self.is_connected():
            return None
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT last_initiated_message FROM users WHERE user_id = %s",
                    (user_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logging.error(f"Error getting last initiated message: {e}")
            return None
        finally:
            self.connection_pool.putconn(conn)
    
    def set_last_initiated_message(self, user_id, date_str):
        """
        Set the last initiated message date for a user
        :param user_id: Telegram user ID
        :param date_str: Date string
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET last_initiated_message = %s WHERE user_id = %s",
                    (date_str, user_id)
                )
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error setting last initiated message: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def add_token_usage(self, user_id, tokens):
        """
        Add token usage for a user
        :param user_id: Telegram user ID
        :param tokens: Number of tokens used
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        today = date.today()
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Check if there's an entry for today
                cursor.execute(
                    "SELECT tokens FROM token_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing entry
                    cursor.execute(
                        "UPDATE token_usage SET tokens = tokens + %s WHERE user_id = %s AND usage_date = %s",
                        (tokens, user_id, today)
                    )
                else:
                    # Create new entry
                    cursor.execute(
                        "INSERT INTO token_usage (user_id, usage_date, tokens) VALUES (%s, %s, %s)",
                        (user_id, today, tokens)
                    )
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding token usage: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def get_current_token_usage(self, user_id):
        """
        Get token usage for today and this month
        :param user_id: Telegram user ID
        :return: Tuple of (today_usage, month_usage)
        """
        if not self.is_connected():
            return 0, 0
        
        today = date.today()
        month = f"{today.year}-{today.month:02d}"
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Get today's usage
                cursor.execute(
                    "SELECT tokens FROM token_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                today_usage = result[0] if result else 0
                
                # Get month usage
                cursor.execute(
                    "SELECT SUM(tokens) FROM token_usage WHERE user_id = %s AND TO_CHAR(usage_date, 'YYYY-MM') = %s",
                    (user_id, month)
                )
                result = cursor.fetchone()
                month_usage = result[0] if result and result[0] else 0
                
                return today_usage, month_usage
        except Exception as e:
            logging.error(f"Error getting token usage: {e}")
            return 0, 0
        finally:
            self.connection_pool.putconn(conn)
    
    def add_image_request(self, user_id, image_size):
        """
        Add image request for a user
        :param user_id: Telegram user ID
        :param image_size: Size of the image ("256x256", "512x512", or "1024x1024")
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        today = date.today()
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Determine which column to update based on image size
                size_column = ""
                if image_size == "256x256":
                    size_column = "size_256"
                elif image_size == "512x512":
                    size_column = "size_512"
                elif image_size == "1024x1024":
                    size_column = "size_1024"
                else:
                    return False
                
                # Check if there's an entry for today
                cursor.execute(
                    f"SELECT {size_column} FROM image_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing entry
                    cursor.execute(
                        f"UPDATE image_usage SET {size_column} = {size_column} + 1 WHERE user_id = %s AND usage_date = %s",
                        (user_id, today)
                    )
                else:
                    # Create new entry with default values
                    size_256 = 1 if size_column == "size_256" else 0
                    size_512 = 1 if size_column == "size_512" else 0
                    size_1024 = 1 if size_column == "size_1024" else 0
                    
                    cursor.execute(
                        "INSERT INTO image_usage (user_id, usage_date, size_256, size_512, size_1024) VALUES (%s, %s, %s, %s, %s)",
                        (user_id, today, size_256, size_512, size_1024)
                    )
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding image request: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def get_current_image_count(self, user_id):
        """
        Get image count for today and this month
        :param user_id: Telegram user ID
        :return: Tuple of (today_count, month_count)
        """
        if not self.is_connected():
            return 0, 0
        
        today = date.today()
        month = f"{today.year}-{today.month:02d}"
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Get today's usage
                cursor.execute(
                    "SELECT size_256 + size_512 + size_1024 FROM image_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                today_count = result[0] if result else 0
                
                # Get month usage
                cursor.execute(
                    "SELECT SUM(size_256 + size_512 + size_1024) FROM image_usage WHERE user_id = %s AND TO_CHAR(usage_date, 'YYYY-MM') = %s",
                    (user_id, month)
                )
                result = cursor.fetchone()
                month_count = result[0] if result and result[0] else 0
                
                return today_count, month_count
        except Exception as e:
            logging.error(f"Error getting image count: {e}")
            return 0, 0
        finally:
            self.connection_pool.putconn(conn)
    
    def add_vision_tokens(self, user_id, tokens):
        """
        Add vision tokens for a user
        :param user_id: Telegram user ID
        :param tokens: Number of tokens used
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        today = date.today()
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Check if there's an entry for today
                cursor.execute(
                    "SELECT tokens FROM vision_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing entry
                    cursor.execute(
                        "UPDATE vision_usage SET tokens = tokens + %s WHERE user_id = %s AND usage_date = %s",
                        (tokens, user_id, today)
                    )
                else:
                    # Create new entry
                    cursor.execute(
                        "INSERT INTO vision_usage (user_id, usage_date, tokens) VALUES (%s, %s, %s)",
                        (user_id, today, tokens)
                    )
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding vision tokens: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def get_current_vision_tokens(self, user_id):
        """
        Get vision tokens for today and this month
        :param user_id: Telegram user ID
        :return: Tuple of (today_tokens, month_tokens)
        """
        if not self.is_connected():
            return 0, 0
        
        today = date.today()
        month = f"{today.year}-{today.month:02d}"
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Get today's usage
                cursor.execute(
                    "SELECT tokens FROM vision_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                today_tokens = result[0] if result else 0
                
                # Get month usage
                cursor.execute(
                    "SELECT SUM(tokens) FROM vision_usage WHERE user_id = %s AND TO_CHAR(usage_date, 'YYYY-MM') = %s",
                    (user_id, month)
                )
                result = cursor.fetchone()
                month_tokens = result[0] if result and result[0] else 0
                
                return today_tokens, month_tokens
        except Exception as e:
            logging.error(f"Error getting vision tokens: {e}")
            return 0, 0
        finally:
            self.connection_pool.putconn(conn)
    
    def add_tts_request(self, user_id, text_length, tts_model):
        """
        Add TTS request for a user
        :param user_id: Telegram user ID
        :param text_length: Length of text converted to speech
        :param tts_model: TTS model used
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        today = date.today()
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Check if there's an entry for today and this model
                cursor.execute(
                    "SELECT characters FROM tts_usage WHERE user_id = %s AND usage_date = %s AND model = %s",
                    (user_id, today, tts_model)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing entry
                    cursor.execute(
                        "UPDATE tts_usage SET characters = characters + %s WHERE user_id = %s AND usage_date = %s AND model = %s",
                        (text_length, user_id, today, tts_model)
                    )
                else:
                    # Create new entry
                    cursor.execute(
                        "INSERT INTO tts_usage (user_id, usage_date, model, characters) VALUES (%s, %s, %s, %s)",
                        (user_id, today, tts_model, text_length)
                    )
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding TTS request: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def get_current_tts_usage(self, user_id):
        """
        Get TTS usage for today and this month
        :param user_id: Telegram user ID
        :return: Tuple of (today_characters, month_characters)
        """
        if not self.is_connected():
            return 0, 0
        
        today = date.today()
        month = f"{today.year}-{today.month:02d}"
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Get today's usage
                cursor.execute(
                    "SELECT SUM(characters) FROM tts_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                today_characters = result[0] if result and result[0] else 0
                
                # Get month usage
                cursor.execute(
                    "SELECT SUM(characters) FROM tts_usage WHERE user_id = %s AND TO_CHAR(usage_date, 'YYYY-MM') = %s",
                    (user_id, month)
                )
                result = cursor.fetchone()
                month_characters = result[0] if result and result[0] else 0
                
                return int(today_characters), int(month_characters)
        except Exception as e:
            logging.error(f"Error getting TTS usage: {e}")
            return 0, 0
        finally:
            self.connection_pool.putconn(conn)
    
    def add_transcription_seconds(self, user_id, seconds):
        """
        Add transcription seconds for a user
        :param user_id: Telegram user ID
        :param seconds: Number of seconds transcribed
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        today = date.today()
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Check if there's an entry for today
                cursor.execute(
                    "SELECT seconds FROM transcription_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing entry
                    cursor.execute(
                        "UPDATE transcription_usage SET seconds = seconds + %s WHERE user_id = %s AND usage_date = %s",
                        (seconds, user_id, today)
                    )
                else:
                    # Create new entry
                    cursor.execute(
                        "INSERT INTO transcription_usage (user_id, usage_date, seconds) VALUES (%s, %s, %s)",
                        (user_id, today, seconds)
                    )
                
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding transcription seconds: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)
    
    def get_current_transcription_duration(self, user_id):
        """
        Get transcription duration for today and this month
        :param user_id: Telegram user ID
        :return: Tuple of (minutes_day, seconds_day, minutes_month, seconds_month)
        """
        if not self.is_connected():
            return 0, 0, 0, 0
        
        today = date.today()
        month = f"{today.year}-{today.month:02d}"
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Get today's usage
                cursor.execute(
                    "SELECT seconds FROM transcription_usage WHERE user_id = %s AND usage_date = %s",
                    (user_id, today)
                )
                result = cursor.fetchone()
                seconds_day = result[0] if result else 0
                
                # Get month usage
                cursor.execute(
                    "SELECT SUM(seconds) FROM transcription_usage WHERE user_id = %s AND TO_CHAR(usage_date, 'YYYY-MM') = %s",
                    (user_id, month)
                )
                result = cursor.fetchone()
                seconds_month = result[0] if result and result[0] else 0
                
                minutes_day, seconds_day = divmod(seconds_day, 60)
                minutes_month, seconds_month = divmod(seconds_month, 60)
                
                return int(minutes_day), round(seconds_day, 2), int(minutes_month), round(seconds_month, 2)
        except Exception as e:
            logging.error(f"Error getting transcription duration: {e}")
            return 0, 0, 0, 0
        finally:
            self.connection_pool.putconn(conn)
    
    def get_current_cost(self, user_id):
        """
        Get current cost for a user
        :param user_id: Telegram user ID
        :return: Dictionary with cost_today, cost_month, and cost_all_time
        """
        if not self.is_connected():
            return {"cost_today": 0.0, "cost_month": 0.0, "cost_all_time": 0.0}
        
        today = date.today()
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT day_cost, month_cost, all_time_cost, last_update FROM cost_tracking WHERE user_id = %s",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    day_cost, month_cost, all_time_cost, last_update = result
                    
                    # Reset day/month cost if needed
                    if last_update != today:
                        if last_update.month != today.month:
                            month_cost = 0.0
                        day_cost = 0.0
                        
                        cursor.execute(
                            "UPDATE cost_tracking SET day_cost = %s, month_cost = %s, last_update = %s WHERE user_id = %s",
                            (day_cost, month_cost, today, user_id)
                        )
                        conn.commit()
                    
                    return {
                        "cost_today": day_cost,
                        "cost_month": month_cost,
                        "cost_all_time": all_time_cost
                    }
                else:
                    return {"cost_today": 0.0, "cost_month": 0.0, "cost_all_time": 0.0}
        except Exception as e:
            logging.error(f"Error getting current cost: {e}")
            return {"cost_today": 0.0, "cost_month": 0.0, "cost_all_time": 0.0}
        finally:
            self.connection_pool.putconn(conn)
    
    def update_cost(self, user_id, day_cost, month_cost, all_time_cost):
        """
        Update cost for a user
        :param user_id: Telegram user ID
        :param day_cost: Cost for today
        :param month_cost: Cost for this month
        :param all_time_cost: All-time cost
        :return: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        today = date.today()
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE cost_tracking SET day_cost = %s, month_cost = %s, all_time_cost = %s, last_update = %s WHERE user_id = %s",
                    (day_cost, month_cost, all_time_cost, today, user_id)
                )
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error updating cost: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

# Create a singleton instance
db_manager = None

def initialize_db_manager(config):
    """
    Initialize the database manager with configuration
    :param config: Dictionary containing database configuration
    """
    global db_manager
    db_manager = DatabaseManager(config)
    return db_manager
