import email
import os
import re
import json
import email
import requests
import torch
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from transformers import BertTokenizerFast, BertForSequenceClassification
from imapclient import IMAPClient
from email.policy import default
from openai import OpenAI
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def connect_to_email_server():
    """
    Connects to Gmail's IMAP service. Once connected, the INBOX folder will be selected
        :param: none
        :return server: server connection (imap.gmail.com, PORT 993 over SSL)
    """
    HOST = 'imap.gmail.com'
    PORT = 993
    USERNAME = 'phishme1212@gmail.com'
    PASSWORD = os.getenv('GMAIL_APP_PASSWORD')  # refer to user manual
    FOLDER = 'INBOX'

    server = IMAPClient(HOST, ssl=True, port=PORT)    
    try:
        server.login(USERNAME, PASSWORD)
        print(f"Sucessfully logged into {USERNAME}")
        server.select_folder(FOLDER)
        print(f"Monitoring {FOLDER} folder")
        return server
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def get_latest_email_content(server):
    """
    Gets the newest email content, email address, and URLs within that email
    :param server: server connection (imap.gmail.com, PORT 993 over SSL)
    :return content: email contents
    :return urls: scraped urls
    :return sender_email: sender's email address
    """
    messages = server.search('ALL')
    content = ""
    urls = []
    sender_email = ""
    # Fetch email
    if messages:
        latest_email_id = messages[-1]
        response = server.fetch([latest_email_id], ['RFC822', 'ENVELOPE'])
        for msgid, data in response.items():
            envelope = data[b'ENVELOPE']
            subject = envelope.subject.decode('utf-8', 'ignore') if envelope.subject else "no subject"
            sender_email = envelope.from_[0].mailbox.decode('utf-8') + "@" + envelope.from_[0].host.decode('utf-8')
            print("Latest email subject:", subject)
            print("Sender's email:", sender_email)

            # Scrape Email HTML
            email_message = email.message_from_bytes(data[b'RFC822'], policy=default)
            content = ""
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == 'text/plain':
                        content = part.get_payload(decode=True).decode('utf-8')
                        break
            else:
                content = email_message.get_payload(decode=True).decode('utf-8')

            # Scrape URL - https://stackoverflow.com/questions/28840908/perfect-regex-for-extracting-url-with-re-findall
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            
            if urls:
                print("URLs were found in email.")
            else:
                print("No URLs were found in email.")

    return content, urls, sender_email, subject

def detect_phishing_openai(email_content):
    """
    Using Open AI Assistants API, uses GPT-4 (a GPT model), to analyze the email and output a confidence score.
    :param email_content: email body content
    :return result: AI Assistant response
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_KEY")) # refer to user manual

    thread = client.beta.threads.create()

    print("OpenAI Assistant Thread ID:", thread.id)
    print("Analyzing email with GPT4.0 from OpenAI...")

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=email_content
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_iEUgRfwm8iq9Thb865UYsKJn"
    )

    timeout = 60  # timeout after 60 seconds
    start_time = time.time()

    while True:
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        for message in messages.data:
            if message.role == "assistant" and message.content:
                return message.content[0].text.value
        
        if (time.time() - start_time) > timeout:
            return "Response timed out. No response from Open AI GPT model."  # better than an IndexError :/

        time.sleep(2)  # wait for 2 seconds before trying again
    
    return "No response from Open AI GPT model." # idk, just in case lol
 
def danger_words(email_content):
    """
    Searches the email content for potential danger words.
    :param email_content: email body content
    :return result: Danger word detected or not.
    """
    # a list of pre-defined "danger" words
    danger_words = [
        "urgent",
        "your account has been suspended",
        "compromised",
        "immediate action required",
        "confidential",
        "verify your account",
        "password reset",
        "click this link",
        "security alert",
        "payment details"
        "refund",
        "charged",
        "fraud",
        "claim a refund",
        "billing department",
        "debit from account",
        "grand total"
    ]

    score = 0  
    print("Searching for danger words...\n")
    email_content_lower = email_content.lower()  
    for word in danger_words:
        if word in email_content_lower:
            # If a danger word is found, then just exit the loop
            score = 1  
            break  
    if score == 1:
        result = "Danger word detected"
    else:
        result = "Danger word not detected"
    return(result)

def load_model_and_tokenizer():
    """
    Loads the LSTM model and Tokenizer.
    :param model: LSTM Model file
    :return tokenizer: Tokenizer file 
    """
    model = load_model('.\\Model Training\\LSTM\\assets\\LSTM_model\\phishing_detection_model_4.h5')
    with open('.\\Model Training\\LSTM\\assets\\LSTM_model\\tokenizer4.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return model, tokenizer

def preprocess_email(email_content, tokenizer):
    """
    Processes the email using the tokenizer, converting words to tokens.
    :param email_content: email body content
    :param tokenizer: Tokenizer file 
    :return padded_sequences: applied padding to sequences of tokens
    """
    sequences = tokenizer.texts_to_sequences([email_content])
    padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    return padded_sequences

def detect_phishing_LSTM(email_content):
    """
    LSTM prediction.
    :param email_content: email body content
    :return final_result: (Sigmoid activation function) float number between 0-1, where 0 is phishing and 1 is safe.
    """
    model, tokenizer = load_model_and_tokenizer()
    processed_email = preprocess_email(email_content, tokenizer)
    print("\nLSTM model is predicting...")
    prediction = model.predict(processed_email)
    
    confidence_score = float(prediction[0][0])
    formatted_confidence_score = "{:.2f}".format(confidence_score)
    
    final_result = formatted_confidence_score
    
    return final_result

def detect_phishing_transformer(email_content):
    """
    BERT Transformer prediction (Training was done on GPU)
    :param email_content: email body content
    :return result: (Softmax activation function) vector of 2 values that sum to 1, giving a probability of safe and phishing email.
    """
    device = torch.device("cpu") # using CPU for prediction, should handle well...
    model_path = ".\\Model Training\\BERT\\assets\\transformer_model" 
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)  

    email_text = email_content

    # tokenize and prepare inputs
    inputs = tokenizer(email_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    # prediction
    print("\nBERT model is predicting...")
    with torch.no_grad():
        outputs = model(**inputs)        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prob_safe, prob_phishing = probabilities[0].tolist()

    return (prob_safe, prob_phishing)

def url_reputation(url):
    """
    Checks for URL Reputation, via APIVoid API
    :param url: URL scraped from email content
    :return result: a dictionary of IP, URL, Risk Score and SSL Certificate Status. Refer to user manual to check for more URL details.
    """
    apivoid_key = os.getenv('API_VOID') # refer to user manual
    url_scrape = f"https://endpoint.apivoid.com/urlrep/v1/pay-as-you-go/?key={apivoid_key}&url={url}"
    response = requests.get(url_scrape)
    json_data_url = response.json()

    if 'error' not in json_data_url:
        print("Analyzing URL with APIVoid...")
        ip = json_data_url['data']['report']['server_details']['ip']
        risk_score = json_data_url['data']['report']['risk_score']['result']
        ssl_certificate = json_data_url['data']['report']['security_checks']['is_valid_https']
    
        result = {
            "IP": ip,
            "URL": url,
            "Risk Score": risk_score,
            "SSL Certificate": "Valid" if ssl_certificate else "Invalid"
        }
    else:
        result = {
            "Error": json_data_url.get('error', 'Unknown error')
        }

    return result

def email_reputation(email):
    """
    Checks for email Reputation, via APIVoid API
    :param email: sender of the email being analyzed
    :return result: a dictionary of Email, Username, Domain, if sus, if spoofable, should block, and Score. Refer to user manual to check for more email details.
    """
    apivoid_key = os.getenv('API_VOID') # refer to user manual
    url_email = f"https://endpoint.apivoid.com/emailverify/v1/pay-as-you-go/?key={apivoid_key}&email={email}"
    response = requests.get(url_email)
    json_data_email = response.json()

    if 'error' not in json_data_email:
        print("Analyzing email with APIVoid...")
        result = {
            "Email": json_data_email["data"]["email"],
            "Username": json_data_email["data"]["username"],
            "Domain": json_data_email["data"]["domain"],
            "Suspicious Username": json_data_email["data"]["suspicious_username"],
            "Suspicious Domain": json_data_email["data"]["suspicious_domain"],
            "Is Spoofable": json_data_email["data"]["is_spoofable"],
            "Recommend Block": json_data_email["data"]["should_block"],
            "Score": json_data_email["data"]["score"]
        }
    else:
        result = {
            "Error": json_data_email.get('error', 'Unknown error')
        }

    return result 

def send_report(sender_email, recipient_email, sender_password, subject, report_content):
    """
    Sends an email report from the sender to the recipient containing the specified subject and content.
    :param sender_email: Email address of the Antiphish Machine
    :param recipient_email: Email address of the recipient
    :param password: Password for the sender's email account, set as an environment variable
    :param subject: Subject line of the report email
    :param report_content: The content of the report to be sent
    """
    server = smtplib.SMTP('smtp-mail.outlook.com', 587)
    server.starttls() # TLS encryption
    
    try:
        server.login(sender_email, sender_password)
        print("Antiphish Machine logged in with:", sender_email)
        
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject
        message.attach(MIMEText(report_content, 'html'))
        
        print("Sending message to:", recipient_email)
        server.send_message(message)
        print("Phishing report was sent successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing server. Continuing to Monitor...")
        print("+------------------------------------------------------------------------------------+")
        server.quit()
    
def report_generator(email_content, sender_subject, phishing_result, danger_result, lstm_result, transformer_results, url_results, email_rep_result, sender_email_address, unsub_results): 
    """
    Sends an email report from the sender to the recipient containing the specified subject and content.
    :param email_content: email body content
    :param sender_subject: Original email's subject
    :param phishing_result: Open AI GPT4.0 model's analysis
    :param danger_result: Danger word detection
    :param lstm_result: Long Short Term Memory model result
    :param transformer_result: BERT model result
    :param url_results: URL reputation results
    :param email_rep_result: Email reputation results
    :param sender_email_addressw: Original sender's email
    Calls send_report() after generating report to sent it to recipient
    """
    # sender_email_address is the email of the sender of the mail being analyzed. sender_email is the Antiphish Machine.
    sender_email = 'report@antiphishmachine.com'
    recipient_email = 'phishme1212@gmail.com'
    sender_password = os.getenv('REPORT_GENERATOR_PASSWORD')  # refer to user manual
    today = datetime.now()

    urls_analysis_html = ''
    for url_result in url_results:
        url_display = url_result.get('URL', 'Error during URL Analysis')  # error handling here because sometimes the URLs may be invalid, (actually may be problems scraping...)
        ip_display = url_result.get('IP', 'N/A')
        risk_score_display = url_result.get('Risk Score', 'N/A')
        ssl_cert_display = 'Valid' if url_result.get('SSL Certificate', False) else 'Invalid'
        urls_analysis_html += f"""
        <tr>
            <td>{url_display}</td>
            <td>{ip_display}</td>
            <td>{risk_score_display}</td>
            <td>{ssl_cert_display}</td>
        </tr>
        """
    
    email_analysis_html = f"""
    <tr>
        <td>{email_rep_result['Email']}</td>
        <td>{email_rep_result['Username']}</td>
        <td>{email_rep_result['Domain']}</td>
        <td>{'Yes' if email_rep_result['Suspicious Username'] else 'No'}</td>
        <td>{'Yes' if email_rep_result['Suspicious Domain'] else 'No'}</td>
        <td>{'Yes' if email_rep_result['Is Spoofable'] else 'No'}</td>
        <td>{'Yes' if email_rep_result['Recommend Block'] else 'No'}</td>
        <td>{email_rep_result['Score']}</td>
    </tr>
    """
     
    lstm_percent = "{:.2%}".format(float(lstm_result))
    lstm_prediction = 'This email is Safe' if float(lstm_result) > 0.5 else 'This email may be a Phishing attempt'
    
    present_time = time.localtime() # this section of code should be refactored to account for user's timezone. Future development may consider pytz library. 
    report_time = time.strftime("%I:%M:%S %p", present_time) 
    report_date = today.strftime('%B %d, %Y')

    # Compiles the report content in HTML format
    subject = f"Phishing Detection Report for {report_date} at {report_time}"
    if unsub_results == True:
        unsub = "Found in Email (Likely Marketing)"
    else:
        unsub = "Not Found in Email (Likely Personal or possible phishing, needs review.)"

    # I don't like this method, but this was the easiest way to generate a beautiful report.
    report_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Detection Report</title>
    <style>
    body {{
        font-family: 'courier', monospace;
        background-color: #1f1f1f;
        color: #333;
        margin: 0;
        padding: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        box-sizing: border-box;
    }}
    .container {{
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        background-color: #353535;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        filter: blur(0.5px);
    }}
    .section {{
        background-color: #b8b8b8;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;;
    }}
    h1 {{
        color: #ffffff;
        text-align: center;
    }}
    p {{
        line-height: 1.6;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }}
    table, th, td {{
        border: 1px solid #747474;
        padding: 8px;
        text-align: left;
    }}
    th {{
        background-color: #8a8a8a;
    }}
    @media (max-width: 600px) {{
        body {{
        padding: 10px;
        }}
        .container {{
        padding: 10px;
        }}
    }}
    </style>
    </head>
    <body>
    <div class="container">
        <h1>Phishing Detection Report</h1>
        <div class="section">
            <h2>Date and Time</h2>
            <p>{report_date} at {report_time}</p>
        </div>
        <div class="section">
            <h2>Email Content:</h2>
            <p><strong>Sender email:</strong> {sender_email_address}</p>
            <p><strong>Subject:</strong> {sender_subject}</p>
            <p>{email_content.replace('<', '&lt;').replace('>', '&gt;')}</p> 
        </div>
        <div class="section">
            <h2>OpenAI Phishing Detection Result:</h2>
            <p>{phishing_result}</p>
        </div>
        <div class="section">
            <h2>Danger Words Detection Result:</h2>
            <h3>This only checks the email for a word that is common in phishing emails, (example: URGENT, COMPROMISED, REFUND, CHARGED). Note: Not finding a danger word does not mean the email is always safe.</h3>
            <p>{danger_result}</p>
        </div>
        <div class="section">
            <h2>Has Option to Unsubscribe:</h2>
            <h3>This only checks if the email has an opt-out option. Note: Finding an unsubscribe option in the email does not mean that the email is always safe.</h3>
            <p>{unsub}</p>
        </div>
        <div class="section">
            <h2>LSTM Model Phishing Prediction:</h2>
            <p>Predition Score: {lstm_percent}</p>
            <p><strong>{lstm_prediction}</strong></p>
        </div>
        <div class="section">
            <h2>Transformer Model Phishing Prediction:</h2>
            <p><strong>Probability of being safe:</strong> {transformer_results[0]:.2%}</p>
            <p><strong>Probability of being phishing:</strong> {transformer_results[1]:.2%}</p>
        </div>
        <div class="section">
            <h2>URLs Analysis Results:</h2>
            <table>
                <tr>
                    <th>URL</th>
                    <th>IP</th>
                    <th>Risk Score</th>
                    <th>SSL Certificate</th>
                </tr>
                {urls_analysis_html}
            </table>
        </div>
        <div class="section">
            <h2>Email Reputation Analysis:</h2>
            <table>
                <tr>
                    <th>Email</th>
                    <th>Username</th>
                    <th>Domain</th>
                    <th>Suspicious Username</th>
                    <th>Suspicious Domain</th>
                    <th>Is Spoofable</th>
                    <th>Recommend Block</th>
                    <th>Score</th>
                </tr>
                {email_analysis_html}
            </table>
        </div>
    </div>
    </body>
    </html>
    """
    print("Generating Report...", subject)

    # Send the email
    send_report(sender_email, recipient_email, sender_password, subject, report_content)

def has_unsubscribe_link(email_content):
    """
    Checks if there is an "Unsubscribe" link in the email content.
    :param email_content: email body content
    :return: True if an "Unsubscribe" link is present, False otherwise.
    """
    unsubscribe_phrases = ["unsubscribe", "opt out", "stop receiving these emails"]
    for phrase in unsubscribe_phrases:
        if phrase in email_content.lower():
            return True
    return False

def idle_mailbox(server):
    """
    Uses IDLE to always monitor for new mail, until process is killed.
    :param server: (imap.gmail.com, PORT 993 over SSL)
    """
    server.idle()
    try:
        while True:
            responses = server.idle_check(timeout=1800)
            if responses:  
                server.idle_done()  
                email_content, urls, sender_email, subject = get_latest_email_content(server)
                if email_content:
                    phishing_result = detect_phishing_openai(email_content)
                    danger_result = danger_words(email_content)
                    LSTM_result = detect_phishing_LSTM(email_content)
                    transformer_result = detect_phishing_transformer(email_content)
                    url_results = [url_reputation(url) for url in urls]
                    email_result = email_reputation(sender_email)
                    unsub_results = has_unsubscribe_link(email_content)
                    report_generator(email_content, subject, phishing_result, danger_result, LSTM_result, transformer_result, url_results, email_result, sender_email, unsub_results)
                server.idle()  
    except KeyboardInterrupt:
        server.idle_done()
    finally:
        server.logout()

def main():
    """
    main function, begin here as entry point.
    """
    server = connect_to_email_server()
    if server is not None:
        try:
            idle_mailbox(server)
        finally:
            server.logout()
    else:
        print("Failed to establish connection.")

main()
