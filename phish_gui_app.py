import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from phish_gui import connect_to_email_server, fetch_email, get_latest_email_content, detect_phishing_openai, danger_words, detect_phishing_LSTM, detect_phishing_transformer, url_reputation, email_reputation

st.title('The AntiPhish Machine')

st.sidebar.header("Options")
check_button = st.sidebar.button('Check Latest Email')
report_button = st.sidebar.button('Only Send Report')
manual_input = st.sidebar.text_area("Paste email content here for manual analysis:")
analyze_button = st.sidebar.button('Analyze Email')

if report_button:
    server = connect_to_email_server()
    if server is not None:
        with st.spinner('Fetching and analyzing the latest email...'):
            fetch_email(server)
            st.success('Done! Check your email for the report from report@antiphishmachine.com.')
            server.logout()
    else:
        st.error('Failed to establish connection to the email server.')

def plot_pie_chart(transformer_results):
    labels = ['Safe', 'Phishing']
    values = transformer_results
    colors = ['lightgreen', 'salmon']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(marker=dict(colors=colors), textinfo='percent+label')
    fig.update_layout(title_text='BERT Prediction')
    return fig

def plot_prediction_bar(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "LSTM Prediction"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "lightslategrey"},
            'steps' : [
                {'range': [0, 0.5], 'color': "salmon"},
                {'range': [0.5, 1], 'color': "lightgreen"}
            ],
            'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))

    return fig

def display_results(email_content, phishing_result, danger_result, lstm_result, transformer_results, url_results, email_rep_result):
    st.subheader("Email Content:")
    st.write(email_content)
    
    st.subheader("OpenAI Phishing Detection Result:")
    st.write(phishing_result)
    
    st.subheader("Danger Words Detection Result:")
    st.write(danger_result)
    
    st.subheader("LSTM Model Phishing Prediction:")
    st.write(f"Prediction Score: {lstm_result}")
    fig = plot_prediction_bar(float(lstm_result))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("BERT (Transformer) Model Phishing Prediction:")
    safe, phishing = transformer_results
    st.write(f"Probability of being safe: {safe:.2%}")
    st.write(f"Probability of being phishing: {phishing:.2%}")
    transformer_results = detect_phishing_transformer(email_content)
    fig = plot_pie_chart(transformer_results)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("URLs Analysis Results:")
    if url_results:
        for url_result in url_results:
            st.write(url_result)
    
    st.subheader("Email Reputation Analysis:")
    st.write(email_rep_result)

if check_button:
    server = connect_to_email_server()
    if server is not None:
        email_content, urls, sender_email, subject = get_latest_email_content(server)
        if email_content:
            phishing_result = detect_phishing_openai(email_content)
            danger_result = danger_words(email_content)
            lstm_result = detect_phishing_LSTM(email_content)
            transformer_results = detect_phishing_transformer(email_content)
            url_results = [url_reputation(url) for url in urls]
            email_rep_result = email_reputation(sender_email)

            display_results(email_content, phishing_result, danger_result, lstm_result, transformer_results, url_results, email_rep_result)

        server.logout()
    else:
        st.error('Failed to establish connection to the email server.')

if analyze_button and manual_input:
    phishing_result = detect_phishing_openai(manual_input)
    danger_result = danger_words(manual_input)
    lstm_result = detect_phishing_LSTM(manual_input)
    transformer_results = detect_phishing_transformer(manual_input)
    url_results = ""
    email_rep_result = ""
    display_results(manual_input, phishing_result, danger_result, lstm_result, transformer_results, url_results, email_rep_result)


st.markdown("""
This program will fetch and analyze the latest email to detect any phishing attempts. Click "Check Latest Email" to view
the email and the analysis without sending a report to the user's email. Or click "Only Send Report" to analyze the email
and only send the report to the user's email. To continuouly monitor the user's email, please run 'phish_monitor.py'           
""")
