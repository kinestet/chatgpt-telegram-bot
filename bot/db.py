from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
from datetime import datetime
import os
from typing import Optional

class User(models.Model):
    """
    Модель пользователя Telegram
    """
    id = fields.BigIntField(pk=True)  # Telegram user_id
    username = fields.CharField(max_length=255, null=True)
    first_name = fields.CharField(max_length=255, null=True)
    last_name = fields.CharField(max_length=255, null=True)
    language_code = fields.CharField(max_length=10, default='en')
    is_active = fields.BooleanField(default=True)  # Активный/пассивный режим
    created_at = fields.DatetimeField(auto_now_add=True)
    last_activity = fields.DatetimeField(auto_now=True)
    settings: fields.ReverseRelation["UserSettings"]

    class Meta:
        table = "users"

class UserSettings(models.Model):
    """
    Настройки пользователя
    """
    id = fields.IntField(pk=True)
    user: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'models.User', related_name='settings'
    )
    auto_message_enabled = fields.BooleanField(default=True)  # Включены ли автосообщения
    auto_message_interval = fields.IntField(default=30)  # Интервал автосообщений в минутах
    auto_message_text = fields.TextField(null=True)  # Текст автосообщения (если null, используется дефолтный)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_settings"

class Chat(models.Model):
    """
    Модель чата (история сообщений)
    """
    id = fields.UUIDField(pk=True)
    user: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'models.User', related_name='chats'
    )
    message = fields.TextField()
    is_bot = fields.BooleanField(default=False)  # True если сообщение от бота
    created_at = fields.DatetimeField(auto_now_add=True)
    tokens_used = fields.IntField(default=0)  # Количество использованных токенов

    class Meta:
        table = "chats"

class Log(models.Model):
    """
    Модель для логирования действий
    """
    id = fields.UUIDField(pk=True)
    user: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'models.User', related_name='logs'
    )
    action = fields.CharField(max_length=255)  # Тип действия (например, 'message', 'command', 'error')
    details = fields.JSONField(null=True)  # Дополнительные детали в JSON
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "logs"

# Pydantic модели для API
User_Pydantic = pydantic_model_creator(User, name="User")
UserSettings_Pydantic = pydantic_model_creator(UserSettings, name="UserSettings")
Chat_Pydantic = pydantic_model_creator(Chat, name="Chat")
Log_Pydantic = pydantic_model_creator(Log, name="Log")

async def init_db():
    """
    Инициализация подключения к базе данных
    """
    await Tortoise.init(
        db_url=os.getenv('DATABASE_URL'),
        modules={'models': ['bot.db']}
    )
    # Создаем таблицы, если их нет
    await Tortoise.generate_schemas()

async def close_db():
    """
    Закрытие подключения к базе данных
    """
    await Tortoise.close_connections()

# Вспомогательные функции для работы с пользователями
async def get_or_create_user(user_id: int, username: Optional[str] = None, 
                           first_name: Optional[str] = None, 
                           last_name: Optional[str] = None,
                           language_code: str = 'en') -> User:
    """
    Получить или создать пользователя
    """
    user, created = await User.get_or_create(
        id=user_id,
        defaults={
            'username': username,
            'first_name': first_name,
            'last_name': last_name,
            'language_code': language_code
        }
    )
    if created:
        # Создаем настройки по умолчанию для нового пользователя
        await UserSettings.create(user=user)
    return user

async def update_user_activity(user_id: int):
    """
    Обновить время последней активности пользователя
    """
    await User.filter(id=user_id).update(last_activity=datetime.utcnow())

async def toggle_auto_messages(user_id: int, enabled: bool) -> bool:
    """
    Включить/выключить автосообщения для пользователя
    """
    settings = await UserSettings.get(user_id=user_id)
    settings.auto_message_enabled = enabled
    await settings.save()
    return enabled

async def set_auto_message_interval(user_id: int, interval_minutes: int) -> int:
    """
    Установить интервал автосообщений
    """
    settings = await UserSettings.get(user_id=user_id)
    settings.auto_message_interval = interval_minutes
    await settings.save()
    return interval_minutes

async def log_action(user_id: int, action: str, details: Optional[dict] = None):
    """
    Записать действие в лог
    """
    await Log.create(
        user_id=user_id,
        action=action,
        details=details
    ) 