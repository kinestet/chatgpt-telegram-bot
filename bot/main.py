import asyncio
import logging
import os
from bot.telegram_bot import ChatGPTTelegramBot
from bot.openai_helper import OpenAIHelper
from bot.db import init_db, close_db

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def main():
    # Проверяем наличие необходимых переменных окружения
    required_env_vars = ['TELEGRAM_BOT_TOKEN', 'OPENAI_API_KEY', 'DATABASE_URL']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Инициализируем базу данных
    try:
        await init_db()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

    try:
        # Создаем экземпляр OpenAIHelper
        openai = OpenAIHelper(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
            presence_penalty=float(os.getenv('OPENAI_PRESENCE_PENALTY', '0.0')),
            frequency_penalty=float(os.getenv('OPENAI_FREQUENCY_PENALTY', '0.0')),
            reply_count=int(os.getenv('OPENAI_REPLY_COUNT', '1')),
            proxy=os.getenv('OPENAI_PROXY', None)
        )

        # Создаем конфигурацию бота
        config = {
            'token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'proxy': os.getenv('TELEGRAM_PROXY', None),
            'bot_language': os.getenv('BOT_LANGUAGE', 'en'),
            'allowed_user_ids': os.getenv('ALLOWED_USER_IDS', '*'),
            'admin_user_ids': os.getenv('ADMIN_USER_IDS', '-'),
            'enable_transcription': os.getenv('ENABLE_TRANSCRIPTION', 'False').lower() == 'true',
            'ignore_group_transcriptions': os.getenv('IGNORE_GROUP_TRANSCRIPTIONS', 'True').lower() == 'true',
            'budget_period': os.getenv('BUDGET_PERIOD', 'monthly'),
            'user_budgets': os.getenv('USER_BUDGETS', '*'),
            'guest_budget': float(os.getenv('GUEST_BUDGET', '100.0')),
            'monthly_budget': float(os.getenv('MONTHLY_BUDGET', '1000.0')),
            'streaming': os.getenv('STREAMING', 'True').lower() == 'true',
            'streaming_timeout': int(os.getenv('STREAMING_TIMEOUT', '60')),
            'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        }

        # Создаем и запускаем бота
        bot = ChatGPTTelegramBot(config, openai)
        await bot.run()

    except Exception as e:
        logging.error(f"Error running bot: {e}")
        raise
    finally:
        # Закрываем соединение с базой данных
        await close_db()

if __name__ == '__main__':
    asyncio.run(main())
