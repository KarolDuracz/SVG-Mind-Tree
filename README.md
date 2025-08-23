<h2>The full description is here</h2>

https://github.com/KarolDuracz/scratchpad/tree/main/Webapp/SVG%20graph%20with%20sequence%20playback

<br />

sqlite3 is part of Python's standard library. But you need to install Flask to run it.
```
pip install Flask
```
Running. Go to the folder where app.py is and simply enter
```
python app.py
```
The server will start on localhost:5000
<br /><br/>
A brief description of what the app does. If you've heard the term "mind maps" or "tree mind maps" that's the idea behind it. Only here, you build a graph dynamically, adding new nodes and changing decisions about how to play the sequence. Everything is saved in the .db database on disk. And if you fill in the TEXT field, you can add any text that will appear next to each node, for example, every 1000 ms (i.e., every 1 second, because that's the default setting, but you can change it in the source code). This is the variable let playIntervalMs = 1000; around line 212. You can create interactive mind maps or similar and replay sequences while seeing descriptions for each step (node) in the corner of the screen. Here's an example of such a decision graph. 


![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/description%20of%20demo.png?raw=true)

If you want to change something, just download it and do it. This is the base code.
