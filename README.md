<div align="center">

# AntiPhish Machine  
### Preventing Phishing Attacks using Deep Learning  
**Project by [Justin Joy](https://github.com/avROCKET)**  

---

[![Watch Demo](https://raw.githubusercontent.com/avROCKET/AntiPhish-Machine/refs/heads/main/Additional%20Resources/logo.png)](https://youtu.be/WruuAY21eGs "AntiPhish Demo")  

[**Watch the Demo Video**](https://youtu.be/WruuAY21eGs)

---

</div>

## Overview  
**AntiPhish Machine** is an intelligent phishing detection system leveraging **deep learning** models — **BERT**, **LSTM**, and **GPT-4 (via OpenAI API)** — to analyze email content and identify potential phishing attempts.  
The system can operate interactively via a GUI or silently in the background, providing automated protection against email threats.

---

## Core Features  

- **Multi-Model Detection** — Compare results from **BERT**, **LSTM**, and **GPT-4** in real time.  
- **Automated Reporting** — Sends analysis reports from `report@antiphishmachine.com`.  
- **User-Friendly GUI** — Analyze emails locally without needing a report sent to email.  
- **Background Monitoring** — Continuously scans inboxes for potential phishing attacks.  

---

## Model Performance  

| Phase | Model | Accuracy |
|:------|:-------|:----------:|
| **Training** | BERT | **89.35%** |
|  | LSTM | **88.64%** |
| **Testing** | BERT | **89.35%** |
|  | LSTM | **90.81%** |

> *All results are from the initial demo training and evaluation runs.*

---

## Setup Instructions  

Clone the repository:
```bash
git clone https://github.com/avROCKET/AntiPhish-Machine.git
cd AntiPhish-Machine
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set required environment variables:
```bash
setx REPORT_GENERATOR_PASSWORD "YOUR_PASSWORD"
setx OPENAI_KEY "YOUR_OPENAI_KEY"
setx GMAIL_APP_PASSWORD "YOUR_APP_PASSWORD"
setx API_VOID "YOUR_API_KEY"
```

---

## Notes & Known Issues  

- The trained `model.safetensors` file (~418 MB) cannot be uploaded due to Git LFS restrictions.  
- API keys and app passwords are **required** for email and AI integration.  
- The user email is **currently hard-coded**; future GUI updates will allow configuration.  

---

## Technology Stack  

| Category | Technologies |
|:----------|:--------------|
| **Machine Learning** | BERT, LSTM, TensorFlow, PyTorch |
| **NLP / AI Integration** | GPT-4 via OpenAI API |
| **Frontend (GUI)** | Tkinter / CustomTkinter |
| **Backend / Email** | Python, smtplib, imaplib |
| **Other Tools** | Pandas, Regex, JSON, OS Environment Variables |

---

## Future Improvements  

- Replace hard-coded email variables with GUI input.  
- Add cloud model deployment for live scanning.  
- Integrate 2FA for report verification.  
- Support for IMAP folders and multiple inboxes.  

---

<div align="center">
  
*“Protecting users from deception, one email at a time.”*  

**© 2025 Justin Joy** • [GitHub](https://github.com/avROCKET) • [Demo Video](https://youtu.be/WruuAY21eGs)

</div>

