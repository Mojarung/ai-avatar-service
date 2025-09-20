#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import asyncio

from loguru import logger

from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
import time
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    GeminiVADParams,
    InputParams
)
from pipecat.services.gemini_multimodal_live.events import (
    StartSensitivity,
    EndSensitivity
)
from pipecat.transcriptions.language import Language
from simli import SimliConfig
from pipecat.services.simli.video import SimliVideoService
from datetime import datetime
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
import aiohttp
import json
from schemas import InterviewResponse
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
# Load environment variables
async def handle_idle(user_idle: UserIdleProcessor, retry_count: int):
    if retry_count == 1:
        message = {"role":"system", "content": "Пользователь молчит. Спросили, пожалуйста, его вежливо, здесь ли он ещё или нет"}
        await user_idle.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
        return True
    if retry_count == 2:
        message = {"role":"system", "content": "Пользователь не отвечает уже долго. Попрощайся, пожалуйста, с ним и заверши интервью"}
        await user_idle.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
        return True
    else:
        return False
# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
datetime_function = FunctionSchema(
    name="get_current_datetime",
    description="Get the current datetime so you can calculate time left for interview",
    properties={},
    required=[]
)
stop_interview = FunctionSchema(
    name="stop_interview",
    description="Stop the interview",
    properties={
        "report": {
            "type": "string",
            "description": "Report of the interview for HR"
        }
    },
    required=["report"]
)
tools = ToolsSchema(standard_tools=[datetime_function, stop_interview])
async def get_current_datetime(params: FunctionCallParams):
    # Fetch weather data from your API
    datetime_data = {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    await params.result_callback(datetime_data)
# Create a tools schema with your functions
def _make_stop_interview(transport: DailyTransport, api_base_url: str, auth_headers: dict, interview_id: int, transcription_ref: dict, task_ref: dict):
    async def _stop_interview(params: FunctionCallParams):
        try:
            args = params.arguments or {}
            report = args.get("report")

            payload = {"end_date": datetime.now().isoformat()}
            if report:
                payload["summary"] = report
            # Attach dialogue transcription if available
            if transcription_ref:
                payload["dialogue"] = transcription_ref
            logger.info(f"Interview {interview_id} stopped with report: {report}")
            async with aiohttp.ClientSession() as session:
                # Save summary/end_date via API
                url = f"{api_base_url}/interviews/{interview_id}"
                async with session.put(url, json=payload, headers=auth_headers) as resp:
                    resp_text = await resp.text()
                    if resp.status >= 400:
                        logger.error(f"Failed to update interview {interview_id}: {resp.status} {resp_text}")
                        await params.result_callback({"ok": False, "status": resp.status, "body": resp_text})
                    else:
                        logger.info(f"Interview {interview_id} updated successfully")
                        # First, give the assistant time to finish the farewell
                        try:
                            task = task_ref.get("task")
                            if task:
                                await task.stop_when_done()
                                logger.info("Pipeline task cancelled")
                        except Exception as e:
                            logger.error(f"Error disconnecting transport: {e}")
                        await params.result_callback({"ok": True})
        except Exception as e:
            logger.exception("Unhandled error in stop_interview")
            try:
                await params.result_callback({"ok": False, "error": str(e)})
            except Exception:
                pass
    return _stop_interview

async def run_bot(interview_id, room_url, token):
    logger.info("Starting bot")
    pipecat_transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Alexandra",
        params=DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=632,
        video_out_height=632,
        # set stop_secs to something roughly similar to the internal setting
        # of the Multimodal Live api, just to align events.
        vad_analyzer=None
    ),
    )
    user_idle = UserIdleProcessor(
        callback=handle_idle,
        timeout=5
    )
    # Configure API base and auth for server-to-server calls
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_token = None
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{api_base_url}/auth/login", json={"username": f"{os.getenv('HR_USERNAME')}", "password": f"{os.getenv('HR_PASSWORD')}"}) as response:
            if response.status == 200:
                api_token = await response.json()
                api_token = api_token["access_token"]
            else:
                logger.error(f"Failed to get API token. Status code: {response.status}")
    auth_headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    interview_data = None
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_base_url}/interviews/{interview_id}", headers=auth_headers) as response:
            if response.status == 200:
                interview_data = await response.json()
                logger.info(f"Interview data: {interview_data}")
            else:
                logger.error(f"Failed to get interview data. Status code: {response.status}")
    interview = InterviewResponse.model_validate(interview_data)
    vacancy = interview.vacancy
    resume = interview.resume
    # Получаем JSON-дружелюбные dict, чтобы корректно сериализовать Enum и datetime
    vacancy_dict = vacancy.model_dump(
        mode='json',
        exclude={"id", "original_url", "creator_id", "hr_id", "auto_interview_enabled", "created_at", "updated_at", "status"}
    )
    resume_dict = resume.model_dump(
        mode='json',
        exclude={"id", "user_id", "vacancy_id", "file_path", "original_filename", "uploaded_at", "processed", "uploaded_by_hr", "hidden_for_hr", "updated_at", "status", "user"}
    )

    vacancy_data = json.dumps(vacancy_dict, ensure_ascii=False, indent=2)
    resume_data = json.dumps(resume_dict, ensure_ascii=False, indent=2)
    
    logger.info(f"Vacancy data: {vacancy_data}")
    logger.info(f"Resume data: {resume_data}")
    system_instruction = f"""
Ты — Александра, продвинутый HR-интервьюер.

**Задача:** Провести структурированное интервью на **русском языке**, соблюдая этические нормы (без дискриминационных вопросов). Твоя роль — оценить кандидата и подготовить отчет для HR-менеджера, **а не принимать решение о найме**.
Текущее время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

### **План Действий**

**1. Внутренний анализ (перед первым вопросом):**
* Выдели из вакансии 5-7 ключевых компетенций.
* Сопоставь их с резюме, определи главные темы для проверки.

**2. Проведение интервью (взаимодействие с кандидатом):**
* **Структура по времени:** Придерживайся плана: Вступление (~5%), Основные вопросы (~70%), Вопросы кандидата (~15%), Завершение (~10%). Следи за тем, чтобы не превышать время интервью(проверяй периодически время).
* **Начало:** Кратко представься и озвучь план беседы.
* **Диалог:** Задавай по **одному** вопросу за раз. Если ответ неполный — задавай уточняющие вопросы.
* **Завершение:** Будь нейтрален. Поблагодари, озвучь следующие шаги (например, «Мы свяжемся с вами в течение N дней») и пожелай хорошего дня. Не давай никаких намеков на решение. (Завершай интервью только после того, как попрощаешься с кандидатом. чтобы отключение не было резким)

**3. ### Итоговый отчёт (для HR-менеджера)

Пиши все числительные словами.

#### Принципы
- Не засчитывай технологию как подтверждённую без конкретных доказательств существенного опыта.
- Оцени глубину строго относительно требований вакансии: если в описании нужен базовый уровень — это допустимый стандарт; если указана ключевая технология — требуй уверенного или экспертного уровня.
- Доказательства глубины должны включать: контекст и масштаб, роль и зону ответственности, инструменты и решения, сложности и как их преодолевали, измеримые результаты.

#### Калибровка глубины
- Базовый: выполняет типовые задачи по инструкциям, ограниченный объём, мало самостоятельных решений.
- Уверенный: самостоятельно проектирует/поддерживает, решает нетривиальные кейсы, объясняет trade‑off‑ы, сравнивает альтернативы.
- Экспертный: принимает архитектурные решения, улучшает метрики, менторит, имеет опыт инцидентов и профилактики.

#### Оценка по компетенциям
Для каждой ключевой компетенции укажи:
- Компетенция: [название]
- Требуемый уровень (по вакансии): [базовый / уверенный / экспертный]
- Уровень кандидата: [базовый / уверенный / экспертный]
- Статус: [подтверждена / частично / не подтверждена]
- Доказательства:
  - Контекст и масштаб: [сфера, размер команды/сервиса, объём данных/нагрузка]
  - Роль и ответственность: [что делал лично, зона ownership]
  - Инструменты и решения: [технологии, почему выбраны, сравнение альтернатив]
  - Сложности: [какие проблемы возникали и как решены]
  - Результаты: [метрики, экономия, скорость, стабильность, качество]
- Разрыв по глубине: [нет / умеренный / существенный] — кратко, чем это критично для роли

#### Критерии статусов
- Подтверждена: есть два‑три конкретных примера из продакшена с чёткой личной ролью, объяснением решений и измеримыми результатами; кандидат демонстрирует понимание принципов и trade‑off‑ов.
- Частично: есть касания или учебные/эпизодические задачи без измеримых результатов или без ясной личной ответственности; ответы поверхностны или фрагментарны.
- Не подтверждена: теоретические ответы без примеров, противоречия или несоответствие требуемому уровню.

#### Сильные стороны
- Укажи две‑три сильные стороны с привязкой к примерам и метрикам.

#### Риски / Зоны роста
- Укажи один‑два ключевых риска, их потенциальное влияние на роль и как их можно компенсировать (онбординг, менторинг, обучение).

#### Рекомендация
- [Рекомендовать / Рассмотреть / Не рекомендовать] — краткая аргументация, строго опираясь на критичные для вакансии компетенции и разрыв по глубине. Если «Рассмотреть», укажи условия: испытательный срок, план развития по конкретным компетенциям.

#### Примечания
- Не ставь «подтверждена» за одиночное касание технологии без примеров масштаба, ответственности и результата.
- Если по вакансии достаточно базового уровня, статус «подтверждена» допустим при базовом уровне кандидата, но обязательно укажи риск при масштабировании задач.
--------------------------------
Пожалуйста, произноси числительные на русском языке, для этого можешь перевести их в письменную форму, например, 3 - "три"
    """
    context = OpenAILLMContext(
    messages=[
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": f"""**Входные данные о кандидате:**
* **Вакансия: {vacancy_data}**
* **Резюме: {resume_data}**
* **Время (минут):5**. Поприветствуй кандидата и начни собеседование."""
        }
    ])
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        voice_id="Aoede",  # Aoede, Charon, Fenrir, Kore, Puck
        language=Language.RU_RU,
        params=InputParams(
            language=Language.RU_RU
        ),
        vad=GeminiVADParams(
                # Чувствительность старта остаётся высокой, чтобы быстро реагировать на пользователя
                start_sensitivity=StartSensitivity.HIGH,
                # Менее агрессивно заканчиваем фразы ассистента, чтобы не обрывать пользователя при паузах
                end_sensitivity=EndSensitivity.LOW,
                # Увеличим подушку перед началом ответа ассистента, чтобы не перебивать короткие паузы пользователя
                prefix_padding_ms=600,
                # Увеличим длительность тишины, требуемую для окончания реплики
                silence_duration_ms=1300,
            ),
        tools=tools,
    )
    context_aggregator = llm.create_context_aggregator(context)
    # Shared transcription object to accumulate dialogue during session
    transcription = {"dialogue": []}
    # Register tools will be done after the pipeline task is created so we can cancel it
    simli = SimliVideoService(
        SimliConfig(
            apiKey=os.getenv("SIMLI_API_KEY"),
            faceId=os.getenv("SIMLI_FACE_ID"),
            handleSilence=True
        ),
        use_turn_server=True,
        latency_interval=0
    )
    transcript = TranscriptProcessor()
    # Build the pipeline
    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            user_idle,
            context_aggregator.user(),
            transcript.user(),
            llm,
            simli,
            pipecat_transport.output(),
            transcript.assistant(),
            context_aggregator.assistant()
        ]
    )

    # Configure the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    # Provide task into the stop_interview closure via a ref dict
    task_ref = {"task": task}
    # Now register functions including stop_interview with access to task
    llm.register_function(
        "get_current_datetime",
        get_current_datetime,
        cancel_on_interruption=True,
    )
    llm.register_function(
        "stop_interview",
        _make_stop_interview(pipecat_transport, api_base_url, auth_headers, interview_id, transcription, task_ref),
        cancel_on_interruption=True,
    )

    # Handle client connection event
    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        await task.queue_frames(
            [
                LLMRunFrame()
            ]
        )
    # transcription already initialized above to ensure availability in stop_interview
    # Handle client disconnection events
    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            transcription["dialogue"].append({"timestamp": msg.timestamp if msg.timestamp else "", "role": msg.role, "content": msg.content})
    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)