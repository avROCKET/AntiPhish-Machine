# Antiphish Machine: Preventing Phishing Attacks using Deep Learning
#### A Project by Justin Joy
&ensp;
&ensp;

#####  [Demo Video Here or Click the Image below](https://youtu.be/WruuAY21eGs): 
[![Demo Video](https://raw.githubusercontent.com/avROCKET/AntiPhish-Machine/refs/heads/main/Additional%20Resources/logo.png)](https://youtu.be/WruuAY21eGs "AntiPhish Demo")

### Features

- Uses BERT/LSTM/GPT-4 using OpenAI API to compare results between models
- Email sent to user from report@antiphishmachine.com
- User is able to use GUI program to see results without report being sent to email
- User is able to let Antiphish Machine monitor the their inbox in the background

------------


### Training Results from initial Demo:
##### BERT
- 89.35% average

##### LSTM
- 88.64% average

------------


### Testing Results from initial Demo:
##### BERT
- 89.35% average

##### LSTM
- 90.81% average


------------


### Bugs, Known Errors, and Other Notes:
- Unable to upload model.safetensors file used in this demo, due to large file size (418MB) and technical errors with Git LFS.
- API keys and App Passwords for email servers required for program to function, you can set them as OS environment variables.

  - `setx REPORT_GENERATOR_PASSWORD “PASSWORD HERE` 
  - `setx OPENAI_KEY “API KEY HERE` 
  - `setx GMAIL_APP_PASSWORD “PASSWORD HERE" `
  - `setx API_VOID “API KEY HERE" `
- At the moment, the user email is hard coded into the program, which can be changed by using the GUI later updates.

