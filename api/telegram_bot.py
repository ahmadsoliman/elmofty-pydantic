import requests
import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def send_reply(chat_id, text):
    """Send a reply message to the user on Telegram."""
    url = "https://api.telegram.org/bot{0}/sendMessage".format(TELEGRAM_BOT_TOKEN)
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        print(f"Error sending reply message: {error}")
        return {"ok": False, "error": str(error)}


INITIAL_MESSAGE = """مرحباً بك! أنا شيخ مسلم متخصص في الإجابة على الأسئلة الدينية وتقديم الفتاوى في مجال المعاملات المالية.

تفضل بطرح سؤالك، وسأبحث في الفتاوى المعتمدة لأقدم لك الإجابة المناسبة وفق الشريعة الإسلامية.

⚠️ تحذير: الردود مولّدة بواسطة الذكاء الاصطناعي وقد تحتوي على أخطاء.
"""

LOADING_MESSAGE = "جاري البحث عن إجابة..."
ERROR_MESSAGE = "عذرا لم أتمكن من الإجابة حاليا،  برجاء المحاولة لاحقا."


def reply_start(chat_id):
    return send_reply(chat_id, INITIAL_MESSAGE)


def reply_loading(chat_id):
    return send_reply(chat_id, LOADING_MESSAGE)


def reply_error(chat_id):
    return send_reply(chat_id, ERROR_MESSAGE)


def delete_loading_message(chat_id, message_id):
    url = "https://api.telegram.org/bot{0}/deleteMessage".format(TELEGRAM_BOT_TOKEN)
    payload = {"chat_id": chat_id, "message_id": message_id}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        print(f"Error deleting message: {error}")
        return {"ok": False, "error": str(error)}
