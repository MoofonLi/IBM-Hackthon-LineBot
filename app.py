from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    MessageAction
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict

# 載入環境變數
load_dotenv()

app = Flask(__name__)

# Line Bot 設定
configuration = Configuration(access_token=os.getenv('CHANNEL_ACCESS_TOKEN'))
line_handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 使用者狀態管理
user_sessions: Dict[str, Dict] = {}

# 指令常數 - 不轉換為小寫，保持原始格式
CONSULTATION_START_COMMANDS = ["start consultation", "開始問診", "start consultation（開始問診）", "Start Consultation（開始問診）"]
CONSULTATION_END_COMMANDS = ["end consultation", "結束問診", "end consultation（結束問診）", "End Consultation（結束問診）"]
POSTOP_START_COMMANDS = ["postoperative care", "術後照護", "postoperative care（術後照護）", "Postoperative Care（術後照護）"]
POSTOP_END_COMMANDS = ["end care", "結束照護", "end care（結束照護）", "End Care（結束照護）"]

def get_or_create_session(user_id: str) -> Dict:
    """建立或取得使用者的對話階段"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "state": "free_chat",
            "consultation_data": [],
            "postop_data": [],
            "timestamp": datetime.now()
        }
    return user_sessions[user_id]

def generate_consultation_form(consultation_data: list) -> str:
    """生成問診表單"""
    if not consultation_data:
        return "No consultation records.\n\n\n無問診紀錄。"
    
    form = "===== Consultation Record =====\n"
    form += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    form += "Content:\n"
    for idx, entry in enumerate(consultation_data, 1):
        form += f"{idx}. Patient: {entry.get('patient', '')}\n"
        if entry.get('response'):
            form += f"   Response: {entry['response']}\n"
    form += "============================\n\n\n"
    
    form += "========= 問診紀錄表 =========\n"
    form += f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    form += "內容：\n"
    for idx, entry in enumerate(consultation_data, 1):
        form += f"{idx}. 病患：{entry.get('patient', '')}\n"
        if entry.get('response'):
            form += f"   回應：{entry['response']}\n"
    form += "============================"
    return form

def handle_free_chat(message: str) -> str:
    """處理自由對話"""
    # Watson 整合的預留位置
    return (
        f"Received your message: {message}\n"
        "Available commands:\n"
        "- Start Consultation\n"
        "- Postoperative Care\n\n\n"
        f"收到您的訊息：{message}\n"
        "可用指令：\n"
        "- 開始問診\n"
        "- 術後照護"
    )

def create_end_consultation_buttons():
    """建立結束問診的快速回覆按鈕"""
    return QuickReply(items=[
        QuickReplyItem(
            action=MessageAction(
                label="End（結束）",
                text="End Consultation（結束問診）"
            )
        )
    ])

def create_end_postop_buttons():
    """建立結束術後照護的快速回覆按鈕"""
    return QuickReply(items=[
        QuickReplyItem(
            action=MessageAction(
                label="End（結束）",
                text="End Care（結束照護）"
            )
        )
    ])

@app.route("/callback", methods=['POST'])
def callback():
    """處理 Line webhook 回調"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@line_handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """處理接收到的訊息"""
    user_id = event.source.user_id
    session = get_or_create_session(user_id)
    received_message = event.message.text  # 不轉換為小寫，保持原始格式

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        # 處理狀態轉換
        if received_message in CONSULTATION_START_COMMANDS:
            session["state"] = "consultation"
            session["consultation_data"] = []
            message = TextMessage(
                text=(
                    "Consultation mode started. Enter 'End Consultation' to finish.\n\n\n"
                    "已開始問診模式。結束時請輸入「結束問診」。"
                ),
                quick_reply=create_end_consultation_buttons()
            )

        elif received_message in CONSULTATION_END_COMMANDS:
            if session["state"] == "consultation":
                form = generate_consultation_form(session["consultation_data"])
                session["state"] = "free_chat"
                message = TextMessage(
                    text=(
                        "Consultation ended. Here's the record:\n\n\n"
                        "問診已結束。以下是紀錄：\n\n" + form
                    )
                )
            else:
                message = TextMessage(
                    text=(
                        "You are not in consultation mode.\n\n\n"
                        "您目前不在問診模式。"
                    )
                )

        elif received_message in POSTOP_START_COMMANDS:
            session["state"] = "postop_care"
            message = TextMessage(
                text=(
                    "Postoperative care mode started. Enter 'End Care' to finish.\n\n\n"
                    "已開始術後照護模式。結束時請輸入「結束照護」。"
                ),
                quick_reply=create_end_postop_buttons()
            )

        elif received_message in POSTOP_END_COMMANDS:
            if session["state"] == "postop_care":
                session["state"] = "free_chat"
                message = TextMessage(
                    text=(
                        "Postoperative care mode ended.\n\n\n"
                        "術後照護模式已結束。"
                    )
                )
            else:
                message = TextMessage(
                    text=(
                        "You are not in postoperative care mode.\n\n\n"
                        "您目前不在術後照護模式。"
                    )
                )

        # 處理不同狀態
        elif session["state"] == "consultation":
            session["consultation_data"].append({
                "patient": received_message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            message = TextMessage(
                text=(
                    "Symptom recorded. Continue or enter 'End Consultation' to finish.\n\n\n"
                    "已記錄您的症狀。請繼續描述或輸入「結束問診」。"
                ),
                quick_reply=create_end_consultation_buttons()
            )

        elif session["state"] == "postop_care":
            # Watson 整合的預留位置
            message = TextMessage(
                text=(
                    f"Postoperative care response: {received_message}\n\n\n"
                    f"術後照護回應：{received_message}"
                ),
                quick_reply=create_end_postop_buttons()
            )

        else:
            message = TextMessage(text=handle_free_chat(received_message))

        # 發送回應
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[message]
            )
        )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)