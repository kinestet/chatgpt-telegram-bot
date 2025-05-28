from __future__ import annotations

import asyncio
import logging
import os
import io
from datetime import datetime, timedelta
from uuid import uuid4

from telegram import BotCommandScopeAllGroupChats, Update, constants, InlineKeyboardButton, InlineKeyboardMarkup
from telegram import InlineQueryResultArticle, InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext, JobQueue

from pydub import AudioSegment
from PIL import Image

from utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files
from openai_helper import OpenAIHelper, localized_text, GPT_ALL_MODELS
from usage_tracker import UsageTracker
from db import get_or_create_user, update_user_activity, toggle_auto_messages, set_auto_message_interval, log_action, Chat as ChatModel


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        bot_language = self.config['bot_language']
        self.commands = [
            BotCommand(command='help', description=localized_text('help_description', bot_language)),
            BotCommand(command='reset', description=localized_text('reset_description', bot_language)),
            BotCommand(command='stats', description=localized_text('stats_description', bot_language)),
            BotCommand(command='resend', description=localized_text('resend_description', bot_language)),
            BotCommand(command='model', description='Change the OpenAI model'),
            BotCommand(command='settings', description='Настройки бота')
        ]

        self.group_commands = [BotCommand(
            command='chat', description=localized_text('chat_description', bot_language)
        )] + self.commands
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}
        self.job_queue = None

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if is_group_chat(update) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        bot_language = self.config['bot_language']
        help_text = (
                localized_text('help_text', bot_language)[0] +
                '\n\n' +
                '\n'.join(commands_description) +
                '\n\n' +
                localized_text('help_text', bot_language)[1] +
                '\n\n' +
                localized_text('help_text', bot_language)[2]
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                     'requested their usage statistics')

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        (transcribe_minutes_today, transcribe_seconds_today, transcribe_minutes_month,
         transcribe_seconds_month) = self.usage[user_id].get_current_transcription_duration()
        current_cost = self.usage[user_id].get_current_cost()

        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, update)
        bot_language = self.config['bot_language']
        
        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            "----------------------------\n"
        )
        
        text_today = (
            f"*{localized_text('usage_today', bot_language)}:*\n"
            f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
            f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
            "----------------------------\n"
        )
        
        text_month = (
            f"*{localized_text('usage_month', bot_language)}:*\n"
            f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
            f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
        )

        # text_budget filled with conditional content
        text_budget = "\n\n"
        budget_period = self.config['budget_period']
        if remaining_budget < float('inf'):
            text_budget += (
                f"{localized_text('stats_budget', bot_language)}"
                f"{localized_text(budget_period, bot_language)}: "
                f"${remaining_budget:.2f}.\n"
            )

        usage_text = text_current_conversation + text_today + text_month + text_budget
        await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            ' is not allowed to resend the message')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                            ' does not have anything to resend')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('resend_failed', self.config['bot_language'])
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('reset_done', self.config['bot_language'])
        )

    async def model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Changes the OpenAI model
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to change the model')
            await self.send_disallowed_message(update, context)
            return

        if not is_admin(self.config, update.message.from_user.id):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not an admin and cannot change the model')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="Only admins can change the model."
            )
            return

        model_name = message_text(update.message).strip()
        if not model_name:
            # Show current model and available models
            current_model = self.config['model']
            available_models = "\n".join([f"- {model}" for model in GPT_ALL_MODELS])
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=f"Current model: {current_model}\n\nAvailable models:\n{available_models}\n\nTo change model, use: /model model_name"
            )
            return

        if model_name not in GPT_ALL_MODELS:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=f"Model '{model_name}' is not available. Use /model to see available models."
            )
            return

        # Update the model in the config
        self.config['model'] = model_name
        self.openai.config['model'] = model_name
        
        # Reset all chat histories to avoid potential issues with different models
        for chat_id in list(self.openai.conversations.keys()):
            self.openai.reset_chat_history(chat_id)
            
        logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                     f'changed the model to {model_name}')
        
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=f"Model changed to {model_name}. All conversations have been reset."
        )

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe voice messages.
        """
        if not self.config.get('enable_transcription', False):
            await update.message.reply_text(
                message_text(update.message.text, 'transcription_disabled'),
                disable_web_page_preview=True
            )
            return

        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to transcribe audio messages')
            await self.send_disallowed_message(update, context)
            return

        if is_group_chat(update) and self.config.get('ignore_group_transcriptions', True):
            logging.info(f'Group chat message, ignoring transcription request')
            return

        logging.info(f'Transcribing voice message by user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        voice = await update.message.voice.get_file()
        voice_ogg = io.BytesIO()
        await voice.download_to_memory(voice_ogg)
        voice_ogg.seek(0)

        try:
            text = await self.openai.transcribe(voice_ogg)
        except Exception as e:
            logging.exception(e)
            await update.message.reply_text(
                message_text(update.message.text, 'transcription_error'),
                reply_to_message_id=get_reply_to_message_id(self.config, update)
            )
            return

        if text:
            # Split into chunks of 4096 characters (Telegram's message limit)
            text_chunks = split_into_chunks(text)
            for text_chunk in text_chunks:
                await update.message.reply_text(
                    text_chunk,
                    reply_to_message_id=get_reply_to_message_id(self.config, update)
                )
        else:
            await update.message.reply_text(
                message_text(update.message.text, 'transcription_error'),
                reply_to_message_id=get_reply_to_message_id(self.config, update)
            )

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Показывает меню настроек бота
        """
        if not await is_allowed(self.config, update, context):
            await self.send_disallowed_message(update, context)
            return

        user = await get_or_create_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            last_name=update.effective_user.last_name,
            language_code=update.effective_user.language_code
        )

        keyboard = [
            [
                InlineKeyboardButton(
                    "✅ Автосообщения включены" if user.settings.auto_message_enabled else "❌ Автосообщения выключены",
                    callback_data="toggle_auto_messages"
                )
            ],
            [
                InlineKeyboardButton(
                    f"⏰ Интервал: {user.settings.auto_message_interval} мин",
                    callback_data="change_interval"
                )
            ]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Настройки бота:\n\n"
            "• Автосообщения: бот будет отправлять сообщения, если вы долго не пишете\n"
            "• Интервал: время ожидания перед отправкой автосообщения",
            reply_markup=reply_markup
        )

    async def settings_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик нажатий на кнопки в меню настроек
        """
        query = update.callback_query
        await query.answer()

        if query.data == "toggle_auto_messages":
            user = await get_or_create_user(user_id=query.from_user.id)
            new_state = await toggle_auto_messages(user.id, not user.settings.auto_message_enabled)
            
            # Обновляем или удаляем задачу в JobQueue
            job_name = f"auto_message_{user.id}"
            if new_state:
                # Добавляем задачу
                context.job_queue.run_repeating(
                    self.send_auto_message,
                    interval=timedelta(minutes=user.settings.auto_message_interval),
                    first=timedelta(minutes=user.settings.auto_message_interval),
                    name=job_name,
                    data={'user_id': user.id}
                )
            else:
                # Удаляем задачу
                current_jobs = context.job_queue.get_jobs_by_name(job_name)
                for job in current_jobs:
                    job.schedule_removal()

            # Обновляем клавиатуру
            keyboard = [
                [
                    InlineKeyboardButton(
                        "✅ Автосообщения включены" if new_state else "❌ Автосообщения выключены",
                        callback_data="toggle_auto_messages"
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"⏰ Интервал: {user.settings.auto_message_interval} мин",
                        callback_data="change_interval"
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_reply_markup(reply_markup=reply_markup)

        elif query.data == "change_interval":
            # Здесь можно добавить логику для изменения интервала
            # Например, показать новое меню с выбором интервала
            intervals = [15, 30, 60, 120]
            keyboard = [
                [InlineKeyboardButton(f"{i} мин", callback_data=f"set_interval_{i}")]
                for i in intervals
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "Выберите интервал автосообщений:",
                reply_markup=reply_markup
            )

        elif query.data.startswith("set_interval_"):
            interval = int(query.data.split("_")[-1])
            user = await get_or_create_user(user_id=query.from_user.id)
            await set_auto_message_interval(user.id, interval)
            
            # Обновляем задачу в JobQueue
            job_name = f"auto_message_{user.id}"
            current_jobs = context.job_queue.get_jobs_by_name(job_name)
            for job in current_jobs:
                job.schedule_removal()
            
            if user.settings.auto_message_enabled:
                context.job_queue.run_repeating(
                    self.send_auto_message,
                    interval=timedelta(minutes=interval),
                    first=timedelta(minutes=interval),
                    name=job_name,
                    data={'user_id': user.id}
                )

            # Возвращаемся к основному меню настроек
            keyboard = [
                [
                    InlineKeyboardButton(
                        "✅ Автосообщения включены" if user.settings.auto_message_enabled else "❌ Автосообщения выключены",
                        callback_data="toggle_auto_messages"
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"⏰ Интервал: {interval} мин",
                        callback_data="change_interval"
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "Настройки бота:\n\n"
                "• Автосообщения: бот будет отправлять сообщения, если вы долго не пишете\n"
                "• Интервал: время ожидания перед отправкой автосообщения",
                reply_markup=reply_markup
            )

    async def send_auto_message(self, context: ContextTypes.DEFAULT_TYPE):
        """
        Отправляет автосообщение пользователю
        """
        job = context.job
        user_id = job.data['user_id']
        
        try:
            user = await get_or_create_user(user_id=user_id)
            if not user.settings.auto_message_enabled:
                return

            # Проверяем, не было ли активности пользователя
            if datetime.utcnow() - user.last_activity < timedelta(minutes=user.settings.auto_message_interval):
                return

            # Отправляем сообщение
            message_text = user.settings.auto_message_text or "Привет! Как твои успехи? Нужна моя помощь?"
            await context.bot.send_message(
                chat_id=user_id,
                text=message_text
            )

            # Логируем действие
            await log_action(
                user_id=user_id,
                action='auto_message_sent',
                details={'message': message_text}
            )

        except Exception as e:
            logging.error(f"Error sending auto message to user {user_id}: {e}")

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обрабатывает текстовые сообщения от пользователя
        """
        if not await is_allowed(self.config, update, context):
            await self.send_disallowed_message(update, context)
            return

        # Получаем или создаем пользователя
        user = await get_or_create_user(
            user_id=update.effective_user.id,
            username=update.effective_user.username,
            first_name=update.effective_user.first_name,
            last_name=update.effective_user.last_name,
            language_code=update.effective_user.language_code
        )

        # Обновляем время последней активности
        await update_user_activity(user.id)

        # Сохраняем сообщение пользователя в базу
        await ChatModel.create(
            user=user,
            message=update.message.text,
            is_bot=False
        )

        # Логируем действие
        await log_action(
            user_id=user.id,
            action='message_received',
            details={'message': update.message.text}
        )

        # Оригинальная логика обработки сообщения
        chat_id = update.effective_chat.id
        self.last_message[chat_id] = update.message.text

        async def _send_message():
            try:
                response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=update.message.text)
                
                # Сохраняем ответ бота в базу
                await ChatModel.create(
                    user=user,
                    message=response,
                    is_bot=True,
                    tokens_used=total_tokens
                )

                # Логируем действие
                await log_action(
                    user_id=user.id,
                    action='message_sent',
                    details={'message': response, 'tokens': total_tokens}
                )

                # Отправляем ответ пользователю
                await update.message.reply_text(
                    response,
                    parse_mode=constants.ParseMode.MARKDOWN
                )

            except Exception as e:
                logging.exception(e)
                await update.message.reply_text(
                    message_text(update.message.text, 'error'),
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _send_message, constants.ChatAction.TYPING)

    def run(self):
        """
        Запускает бота
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .build()

        # Сохраняем job_queue для использования в других методах
        self.job_queue = application.job_queue

        # Добавляем обработчики команд
        if len(self.config['admin_user_ids'].split(',')) > 0 and self.config['admin_user_ids'] != '-':
            application.add_handler(CommandHandler('stats', self.stats, filters=ChatType.PRIVATE))
            application.add_handler(CommandHandler('allow', self.allow, filters=ChatType.PRIVATE))
            application.add_handler(CommandHandler('disallow', self.disallow, filters=ChatType.PRIVATE))
            application.add_handler(CommandHandler('broadcast', self.broadcast, filters=ChatType.PRIVATE))
            application.add_handler(CommandHandler('block', self.block, filters=ChatType.PRIVATE))
            application.add_handler(CommandHandler('unblock', self.unblock, filters=ChatType.PRIVATE))

        application.add_handler(CommandHandler('start', self.start))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('resend', self.resend))
        application.add_handler(CommandHandler('model', self.model))
        application.add_handler(CommandHandler('settings', self.settings))

        # Добавляем обработчик для кнопок настроек
        application.add_handler(CallbackQueryHandler(self.settings_callback, pattern="^(toggle_auto_messages|change_interval|set_interval_)"))

        application.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, self.transcribe))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.prompt))
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[ChatType.PRIVATE]))

        application.add_error_handler(error_handler)

        # Запускаем бота
        application.run_polling()

