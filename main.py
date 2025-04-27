import streamlit as st
import imaplib
import email
from email.header import decode_header
import pickle
import pandas as pd
import scipy.sparse as sp
from twilio.rest import Client
from transformers import pipeline
import textwrap
import base64
from PIL import Image

# ========== Background Image Full Screen ==========
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .css-1d391kg {{
        background: rgba(255, 255, 255, 0.7);  /* Semi-transparent background for selection box */
        padding: 20px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set your uploaded image path for the background
set_background("whatsapp-gmail.png")

# ========== Load Models ==========
with open("email_priority_model.pkl", "rb") as file:
    loaded_model, vectorizer, label_encoder = pickle.load(file)

reply_generator = pipeline("text-generation", model="gpt2")

# ========== Functions ==========
def predict_email_priority(email_text):
    email_tfidf = vectorizer.transform([email_text])
    email_length = len(email_text)
    email_features = sp.hstack((email_tfidf, [[email_length]]))
    prediction = loaded_model.predict(email_features)[0]
    priority_label = label_encoder.inverse_transform([prediction])[0]
    return priority_label

def generate_reply_suggestion_email(email_subject):
    response = reply_generator(f"Generate a professional reply for: {email_subject}", max_length=50, num_return_sequences=1, truncation=True)
    return response[0]["generated_text"]

def send_whatsapp_reminder(to_phone_number, subject, priority, reply_suggestion):
    TWILIO_ACCOUNT_SID = ""
    TWILIO_AUTH_TOKEN = ""
    TWILIO_WHATSAPP_NUMBER = ""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message_body = f"Reminder: Email regarding '{subject}' is marked as {priority}. Suggested Reply: {reply_suggestion}"
    try:
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f"whatsapp:{to_phone_number}"
        )
        st.success(f"✅ WhatsApp Reminder sent to {to_phone_number}.")
    except Exception:
        st.error("❌ Error sending WhatsApp message.")

def generate_whatsapp_chat():
    chat = """
    [10:00 AM] Alice: Hey everyone, I need some help with digital marketing. Any suggestions?
    [10:02 AM] Bob: Hi Alice! Are you looking for organic growth or paid ads?
    [10:03 AM] Charlie: I’ve tried both, and a mix of both works best.
    [10:05 AM] Alice: Mainly social media growth, but I’m open to ads if necessary.
    [10:07 AM] Diana: I’ve used XYZ Agency before. They offer different packages based on your budget.
    [10:08 AM] Ethan: I second that! XYZ Agency has some great plans.
    [10:10 AM] Frank: I can recommend ABC Marketing. They have good engagement strategies.
    [10:12 AM] George: Alice, what’s your budget like? That might help us recommend better.
    [10:14 AM] Alice: I’m willing to invest around $500 per month if it guarantees good engagement.
    [10:15 AM] Bob: Here’s what I know about XYZ Agency:
    1. Basic ($299/month) – Social media management & content creation.
    2. Standard ($599/month) – Includes ads management & analytics.
    3. Premium ($999/month) – Covers full-scale marketing, SEO, and lead generation.
    [10:18 AM] Henry: I’ve heard good things about their Standard plan. Seems like a good fit for you.
    [10:20 AM] Ian: The Basic plan is good for consistency, but the Standard plan helps if you want faster engagement with ads.
    [10:22 AM] Jack: I agree with Ian. If growth is your priority, the Standard plan should be your best bet.
    [10:24 AM] Alice: Got it! I’ll consider the Standard plan then. Thanks, everyone!
    """
    return chat.strip()

def summarize_chat_with_lemato(chat):
    summary = "LEMATO SUMMARY: Alice asked for digital marketing help. Bob, Charlie, Diana, Ethan, Frank, George, Henry, Ian, and Jack provided recommendations. XYZ Agency was suggested, and its pricing was discussed. Alice decided to go with the Standard plan for better growth."
    return summary

def generate_reply_suggestion_whatsapp():
    return "The Standard plan includes ad strategies that can help grow your social media faster. Would you like a free consultation to see how it fits your goals?"

# ========== Main Streamlit App ==========
st.title("📧📱 Personal Communication Assistant")

# Set the sidebar image (ensure the image path is correct)
image_path = "C:/Users/DELL/Desktop/ass project/ss.png"
st.sidebar.image(image_path, use_column_width=True)

option = st.sidebar.selectbox("Choose Assistant:", ["Email Assistant", "WhatsApp Analyzer"])

# ================= EMAIL ASSISTANT =================
if option == "Email Assistant":
    st.header("📬 Email Priority Predictor & WhatsApp Notifier")

    EMAIL_ACCOUNT = st.text_input("Enter your email ID:")
    EMAIL_PASSWORD = st.text_input("Enter your email password or app password:", type="password")
    PHONE_NUMBER = st.text_input("Enter your WhatsApp number (with country code, e.g., +919876543210):")

    if st.button("Fetch Emails & Send Notifications"):
        if EMAIL_ACCOUNT and EMAIL_PASSWORD and PHONE_NUMBER:
            with st.spinner('🔄 Fetching emails and sending WhatsApp reminders...'):
                try:
                    mail = imaplib.IMAP4_SSL("imap.gmail.com")
                    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
                    mail.select("inbox")
                    
                    status, messages = mail.search(None, "ALL")
                    email_ids = messages[0].split()

                    emails_data = []
                    for email_id in email_ids[-5:]:
                        status, msg_data = mail.fetch(email_id, "(RFC822)")
                        for response_part in msg_data:
                            if isinstance(response_part, tuple):
                                try:
                                    msg = email.message_from_bytes(response_part[1])
                                    subject, encoding = decode_header(msg["Subject"])[0]
                                    subject = subject.decode(encoding) if isinstance(subject, bytes) else subject
                                    sender = msg.get("From")
                                    priority = predict_email_priority(subject)
                                    reply_suggestion = generate_reply_suggestion_email(subject)
                                    emails_data.append([sender, subject, priority, reply_suggestion])

                                    # Display on screen
                                    st.write(f"📨 From: {sender}\nSubject: {subject}\nPriority: {priority}\nSuggested Reply: {reply_suggestion}")

                                    # Send WhatsApp Reminder
                                    send_whatsapp_reminder(PHONE_NUMBER, subject, priority, reply_suggestion)
                                except Exception as e:
                                    st.error(f"⚠️ Skipping email due to error: {e}")

                    # Save as Excel
                    emails_df = pd.DataFrame(emails_data, columns=["Sender", "Subject", "Priority", "Reply Suggestion"])
                    emails_df.to_excel("email_priorities.xlsx", index=False)
                    st.success("✅ Email priorities saved to 'email_priorities.xlsx'.")

                    # Download Button
                    with open("email_priorities.xlsx", "rb") as f:
                        st.download_button(
                            label="📥 Download Email Priorities Excel",
                            data=f,
                            file_name="email_priorities.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                except Exception as e:
                    st.error(f"❌ Error: {e}")

                finally:
                    try:
                        mail.logout()
                    except:
                        pass
        else:
            st.warning("⚠️ Please fill all fields.")

# ================= WHATSAPP ANALYZER =================
elif option == "WhatsApp Analyzer":
    st.header("💬 WhatsApp Business Chat Analyzer")

    if st.button("Generate and Analyze Chat"):
        with st.spinner('🛠️ Generating and summarizing chat...'):
            chat = generate_whatsapp_chat()
            summary = summarize_chat_with_lemato(chat)
            reply_suggestion = generate_reply_suggestion_whatsapp()

            st.text_area("📜 WhatsApp Chat:", chat, height=300)
            st.text_area("📝 Summarized Chat:", summary, height=120)
            st.text_area("💬 Reply Suggestion:", reply_suggestion, height=80)
