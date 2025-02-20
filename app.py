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
from watsonx import WatsonX, Document
from prompts import POSTOP_CARE_PROMPT, QUESTIONNAIRE_PROMPT, GENERAL_QUERY_PROMPT

# 載入環境變數
load_dotenv()

app = Flask(__name__)

watsonx = WatsonX()

# Line Bot 設定
configuration = Configuration(access_token=os.getenv('CHANNEL_ACCESS_TOKEN'))
line_handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 使用者狀態管理
user_sessions: Dict[str, Dict] = {}

# 指令常數 - 不轉換為小寫，保持原始格式
QUESTIONNAIRE_START_COMMANDS = ["start questionnaire", "開始問診", "start questionnaire（開始問診）", "Questionnaire of Health (健康情況問卷)"]
QUESTIONNAIRE_END_COMMANDS = ["end questionnaire", "結束問診", "end questionnaire（結束問診）", "End Questionnaire（結束問診）"]
POSTOP_START_COMMANDS = ["postoperative care", "術後照護", "postoperative care（術後照護）", "Postoperative Care（術後照護）"]
POSTOP_END_COMMANDS = ["end care", "結束照護", "end care（結束照護）", "End Care（結束照護）"]

def get_or_create_session(user_id: str) -> Dict:
    """建立或取得使用者的對話階段"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "state": "free_chat",
            "questionnaire_data": [],
            "postop_data": [],
            "timestamp": datetime.now()
        }
    return user_sessions[user_id]

def generate_questionnaire_form(questionnaire_data: list) -> str:
    """生成問診表單"""
    if not questionnaire_data:
        return "No questionnaire records.\n\n\n無問診紀錄。"
    
    form = "===== Questionnaire Record =====\n"
    form += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    form += "Content:\n"
    for idx, entry in enumerate(questionnaire_data, 1):
        form += f"{idx}. Patient: {entry.get('patient', '')}\n"
        if entry.get('response'):
            form += f"   Response: {entry['response']}\n"
    form += "============================\n\n\n"
    
    form += "========= 問診紀錄表 =========\n"
    form += f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    form += "內容：\n"
    for idx, entry in enumerate(questionnaire_data, 1):
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
        "- Start Questionnaire\n"
        "- Postoperative Care\n\n\n"
        f"收到您的訊息：{message}\n"
        "可用指令：\n"
        "- 開始問診\n"
        "- 術後照護"
    )

def create_end_questionnaire_buttons():
    """建立結束問診的快速回覆按鈕"""
    return QuickReply(items=[
        QuickReplyItem(
            action=MessageAction(
                label="End（結束）",
                text="End Questionnaire（結束問診）"
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

def load_documents():
    docs = {}
    try:
        for filename in ['questionnaire.txt', 'general.txt', 'postop.txt']:
            file_path = os.path.join('docs', filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {filename} not found")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        docs[filename] = Document(content=content, metadata={'source': filename})
                        watsonx.process_document(content)
                    else:
                        print(f"Warning: Empty content in {filename}")
            except UnicodeDecodeError:
                print(f"Error: Unable to decode {filename} with UTF-8 encoding")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
    except Exception as e:
        print(f"Error in load_documents: {str(e)}")
        
    return docs


# 載入文檔
DOCUMENTS = load_documents()

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
    user_id = event.source.user_id
    session = get_or_create_session(user_id)
    received_message = event.message.text

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        # 首先檢查是否為開始命令
        if received_message in POSTOP_START_COMMANDS:
            session["state"] = "postop_care"
            message = TextMessage(
                text="開始術後照護對話。請描述您的狀況或提出問題。",
                quick_reply=create_end_postop_buttons()
            )
            
        elif received_message in QUESTIONNAIRE_START_COMMANDS:
            session["state"] = "questionnaire"
            message = TextMessage(
                text="開始問診。請描述您的症狀。",
                quick_reply=create_end_questionnaire_buttons()
            )
            
        # 再檢查是否為結束指令
        elif received_message in QUESTIONNAIRE_END_COMMANDS and session["state"] == "questionnaire":
            form = generate_questionnaire_form(session["questionnaire_data"])
            session["state"] = "free_chat"
            session["questionnaire_data"] = []
            message = TextMessage(text=f"問診已結束。已生成記錄表：\n{form}")
            
        elif received_message in POSTOP_END_COMMANDS and session["state"] == "postop_care":
            session["state"] = "free_chat"
            session["postop_data"] = []
            message = TextMessage(text="術後照護對話已結束。")
            
        # 最後根據當前狀態處理一般對話
        elif session["state"] == "postop_care":
            context = watsonx.find_relevant_context(received_message)
            response = watsonx.generate_response(
                context=context,
                user_input=received_message,
                prompt_template=POSTOP_CARE_PROMPT
            )
            message = TextMessage(
                text=response,
                quick_reply=create_end_postop_buttons()
            )
            
        elif session["state"] == "questionnaire":
            context = watsonx.find_relevant_context(received_message)
            response = watsonx.generate_response(
                context=context,
                user_input=received_message,
                prompt_template=QUESTIONNAIRE_PROMPT
            )
            session["questionnaire_data"].append({
                "patient": received_message,
                "response": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            message = TextMessage(
                text=response,
                quick_reply=create_end_questionnaire_buttons()
            )
            
        else:  # free_chat 狀態
            # 檢查是否為命令文字，如果不是則使用一般回應
            if received_message in POSTOP_START_COMMANDS or received_message in QUESTIONNAIRE_START_COMMANDS:
                context = watsonx.find_relevant_context(received_message)
                response = watsonx.generate_response(
                    context=context,
                    user_input=received_message,
                    prompt_template=GENERAL_QUERY_PROMPT
                )
                message = TextMessage(text=response)
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