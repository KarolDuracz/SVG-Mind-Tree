> [!WARNING]  
> I don't have much experience using Whisper or building browser extensions. Especially when it comes to sound settings in systems like Windows 8.1 <> Windows 10/11. There are also settings for the system's microphone, sound card, inputs, and so on. Furthermore, from what I've learned, browser extensions in newer systems have different functions and methods. So, there's no guarantee it will work on your system. This is just a demo to show how it can be solved this way.

<h2>This is for informational purposes only. This requires deeper investigation. For tests only.</h2>

<h2>Chrome extension + Whisper OpenAI</h2>

Whisper : https://github.com/openai/whisper
<br /><br />
After downloading these files, the first step is to load extension into your browser.
1. You need to set developer mode, as shown in image on the right. The slider confirms that mode is active.
2. Then, on the left, there's a "load unpacked" button. Click on it.
3. As you can see, in my case, the extension is somewhere in /Documents/progs/extensions/ - as shown in the image, navigate to this folder and click "Select folder".

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Voice%20Commands%20Interface/images/load%20extension%20to%20chrome.png?raw=true)

If everything is OK, it has loaded. You should see extension as in the image below. Pressing "Start Listening" for the first time triggers device detection and asks for permission to use microphone. An additional window will open asking for permission. Confirm. Then, if the system is running, it should detect sound from the microphone in the background and automatically start recording. Then, if there is no sound after 2 seconds, it stops recording and saves file to disk. It runs in the background constantly, automatically detecting sound and starting recording. So, be careful when testing. Do not leave this extension enabled.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Voice%20Commands%20Interface/images/in%20work.png?raw=true)

Here you can see the path to CURL and command to server where the Whisper Base Model runs and recognizes speech and processes it into text. Extension has a path set to save under the path.

```
C:\Users\username\Downloads\whisper
```

Go to the folder where server.py is. It's right here, along with the files for the extension, which is here.

```
C:\Users\username\Documents\progs\extensions\whisper-ext - options
```

Start server. In this image, you can see the command "python -m http.server" running on port 8000. This isn't for this step. It's in the image below. But this server is in a different CMD window ( you don't see it here on this image ), and curl sends requests to localhost:5000 because that's how the Whisper server runs.

```
python server.py
```

And as you can see in the background notepad, the command to the server with curl looks like this

```
curl -F file=@/path/to/sample.webm http://localhost:5000/transcribe
```

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Voice%20Commands%20Interface/images/whisper%20server%20commands%20transcription.png?raw=true)

This is part for this right cmd windows with "python -m http.server" running on port 8000. It might not be necessary to run this, but for file handling purposes, it probably is (?). I've placed "read_webm_audio.html" here, which is currently running on this server. And that's what you see in the image below. There's also a file called recording-2025-08-24-14-40-03.webm, which is being recorded for test purposes. In the last line of the CMD (image above), WHISPER has converted the text from the Google Translator voiceover into English text.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Voice%20Commands%20Interface/images/read%20webm%20audio.png?raw=true)

