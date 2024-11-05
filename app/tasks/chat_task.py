from app.core.celery_app import celery_app


@celery_app.task
def process_chat_message(content: str):
    print(f"Processing message: {content}")
