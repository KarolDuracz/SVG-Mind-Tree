> [!NOTE]  
>  MLP that Andrej gave to learn, how to train network on text, that can pack and extract patterns from little database of names. It can learn patterns from 32,000 names and reproduce them. GPT-2 and Transformer look like they were trying to build a much larger model that does the same thing ( pack all the text from the internet ) + understands text. It looks like they tried to build a much larger model that learns patterns from text in the same way back in 2017. Now is GPT-5, is amazing. Amazing work by the engineering and all the other experts behind it. And the work ethic of OpenAI, the organization, the way it solves problems, etc. (...) Today, I can't work like them, I can't solve problems like them, be organized like them, etc. You can learn from people like them. So for now, I'll leave this topic with "tiny model." To do that, I need to learn better PyTorch documentation, python, and stuff like that to solve this. So it will take me a while to get back to managing this repo. IT IS CLOSED FOR NOW


> [!WARNING]
> This isn't ML/AI/AGI stuff. I just need it to build a simple application for generating football tactics. Synthetic data. The concept of AI/AGI is more than that. Perhaps it's "judgement" between moves, something that "sees" the future? It knows what will happen based on predictions. True AI is something deeper. (...) This part about the large language model is just my part of the learning. I am trying this approach because it gives greater opportunities to learn how the network works, each element. Instead of duplicating what is already there. You need to think about the problem again this way. This may be confusing so I'd like to clarify.

> [!TIP]
> Update: Go to the version_2 folder here. There's a more user-friendly version there. You can simply turn on drawing mode and draw on the canvas instead of clicking or sending terminal queries. The same goes for changing the target point for the entire path.

> [!NOTE]  
> Please read the "Updates" section at the bottom of this page. I've explained why I use the tokenizer with GPT, etc.

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
Server will start on localhost:5000
<br /><br/>
A brief description of what app does. If you've heard term "mind maps" or "tree mind maps" that's the idea behind it. Only here, you build a graph dynamically, adding new nodes and changing decisions about how to play sequence. Everything is saved in .db database on disk. And if you fill in TEXT field, you can add any text that will appear next to each node, for example, every 1000 ms (i.e., every 1 second, because that's the default setting, but you can change it in source code). This is variable "let playIntervalMs = 1000;" around line 212 in index.html. You can create interactive mind maps or similar and replay sequences while seeing descriptions for each step (node) in the corner of the screen. Here's an example of such a decision graph. 


![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/description%20of%20demo.png?raw=true)

If you want to change something, just download it and do it. This is base code.
<h2>Some sample queries using CURL tool</h2>
curl https://curl.se/
<br /><br />

Base URLs

```
UI Pages:

http://localhost:5000/ → serves index.html

http://localhost:5000/admin → serves admin.html

API Root: http://localhost:5000/api/...
```

ADMIN : # Clear all nodes and connections

```
curl -X POST http://localhost:5000/api/admin/clear
```

NODES : # Get all nodes

```
curl http://localhost:5000/api/nodes
```

NODES: # Get a specific node (with connections) ID 1 for example

```
curl http://localhost:5000/api/nodes/1
```

NODES : # Create a node

```
curl -X POST http://localhost:5000/api/nodes \
     -H "Content-Type: application/json" \
     -d '{
           "name": "My Node",
           "description": "Some description",
           "text": "Long text here"
         }'
```

NODES : # Create and connect to an existing node ( create new branch from existing #1 )

```
curl -X POST http://localhost:5000/api/nodes \
     -H "Content-Type: application/json" \
     -d '{
           "name": "Child Node",
           "connect_to": 1,
           "sequence": true
         }'

```

NODES: # Update a node ( update name and location for a node ID 1 )

```
curl -X PUT http://localhost:5000/api/nodes/1 \
     -H "Content-Type: application/json" \
     -d '{
           "name": "Updated Name",
           "x": 200,
           "y": 100
         }'
```

NODES : # Delete a node ID 1

```
curl -X DELETE http://localhost:5000/api/nodes/1
```

CONNECTIONS : # Get all connections

```
curl http://localhost:5000/api/connections
```

CONNECTIONS : # Create a connection ( ID 1 to ID 2 )

```
curl -X POST http://localhost:5000/api/connections \
     -H "Content-Type: application/json" \
     -d '{
           "source": 1,
           "target": 2,
           "metadata": "manual link"
         }'
```

CONNECTIONS : Delete a connection ( delete ID 1 )

```
curl -X DELETE http://localhost:5000/api/connections/1
```

Add random nodes and random connections

```
curl -X POST http://localhost:5000/api/generate_random \
     -H "Content-Type: application/json" \
     -d '{"count": 5}'
```

<h3>These are sample queries, now version for WINDOWS -> CMD and Powershell</h3>

CMD style with \" and ^ for multiline

```
curl -X POST http://localhost:5000/api/nodes ^
  -H "Content-Type: application/json" ^
  -d "{\"name\": \"My Node\", \"description\": \"Some description\", \"text\": \"Long text here\"}"
```

Powershell style 

```
curl -X POST http://localhost:5000/api/nodes `
  -H "Content-Type: application/json" `
  -d '{"name": "My Node", "description": "Some description", "text": "Long text here"}'
```

<h3>Some examples for CMD</h3>

- new node and connection ( create new branch from existing #13 )

```
curl -X POST http://localhost:5000/api/nodes ^
     -H "Content-Type: application/json" ^
     -d "{\"name\": \"#14 My Node\",\"connect_to\": 13, \"sequence\": \"true\"}"
```

- connect together existing #13 and #14

```
curl -X POST http://localhost:5000/api/connections ^
     -H "Content-Type: application/json" ^
     -d "{\"source\": 13,\"target\": 14,\"metadata\": \"manual link\"}"
```

- update name and location for a node ID 1

```
curl -X PUT http://localhost:5000/api/nodes/1 ^
     -H "Content-Type: application/json" ^
     -d "{ \"name\": \"new name and location test\",\"x\": 200,\"y\": 100}"
```

- update ( edit ) description field to change playback direction for a node ID 13 to 17

```
curl -X PUT http://localhost:5000/api/nodes/13 ^
     -H "Content-Type: application/json" ^
     -d ""{\"description\":\"#17\"}"
```

- update ( edit ) existing node ID 3 and change name, text field and location to (400, 400)

```
curl -X PUT http://localhost:5000/api/nodes/3 ^
     -H "Content-Type: application/json" ^
     -d "{ \"name\": \"new name for ID 3\", \"text\": \" abcd text text\", \"x\": 400,\"y\": 400}"
```

<h3>Real life scenario</h3>

1. Scenario if you want to change direction but you don't have a connection created yet (i.e. this sets connection between #14 and #4)

```
curl -X PUT http://localhost:5000/api/nodes/14 -H "Content-Type: application/json" -d "{\"description\":\"#4\"}"
```

2. but if you don't have a connection yet, you need to create one first

```
curl -X POST http://localhost:5000/api/connections ^
     -H "Content-Type: application/json" ^
     -d "{\"source\": 14,\"target\": 4,\"metadata\": \"manual link\"}"
```

3. now you can change direction (update description field for #ID 14 for direction to #4)

```
curl -X PUT http://localhost:5000/api/nodes/14 -H "Content-Type: application/json" -d "{\"description\":\"#4\"}"
```

4. if the playback does not play in direction #14, change the previous node and playback path (in my case it is e.g. node ID 13)

```
curl -X PUT http://localhost:5000/api/nodes/13 -H "Content-Type: application/json" -d "{\"description\":\"#14\"}"
```

REMEMBER TO ADD # before ID {\"description\":\"#14\"}

<h3>Real life scenario</h3>

1. A few commands for windows cmd that can quickly list whether the descriptions fields have correct names according to pattern "# + DIGITS"

```
# This will show you a general overview of what's in the descriptions fields.
# This can help you see the general overview of the descriptions values. There may be errors, such as missing #, etc.
curl http://localhost:5000/api/nodes | findstr /i "description"

# There are two arguments here: "descriptions" and "id." You'll see a slightly larger log with the ID.
curl http://localhost:5000/api/nodes | findstr /i "description id"

# Displays only descriptions fields, only with values ​​that have # + 2 digits.
curl http://localhost:5000/api/nodes | findstr /ri "description.*#[0-9][0-9]*"

# Check if a specific ID is present in the list, e.g., #23 - if present, return occurrences (?); if absent, return 0.
curl http://localhost:5000/api/nodes | findstr /i "description" | findstr "#23" | find /c /v ""

# Count log lines from two arguments, "description id"
curl http://localhost:5000/api/nodes | findstr /i "description id" | find /c /v ""
```

For example, in a scenario where the first node created in node plan has ID #4, the root node. Subsequent root nodes were created and other nodes were connected to root ID 4. Now, ID 4 is not starting node that starts the playback sequence. Only the nodes that were connected to it. This can be verified by knowing root ID, for example.

1. Displays information about Node #4
   
```
# all information about node
curl http://localhost:5000/api/nodes/4
```

2. The pattern is Source -> Target. So, if ID4 has a connection as TARGET from another node, it means it lies somewhere in sequence. So this shorter listing will show connection ID, source ID, and target ID. You can also check how many connections there are as TARGET to, for example, ID #4. Simply by eliminating this specific node, like this 4.

```
# only "id source target" extract from informations
C:\curl\curl-8.15.0_4-win64-mingw\bin>curl http://localhost:5000/api/nodes/4 | findstr /i "id source target"

# get number of lines for results for this command -> looking for pattern "target" : 4
curl http://localhost:5000/api/nodes/4 | findstr /r "^[ ]*\"target\": 4$" | find /c /v ""

# all matching results
curl http://localhost:5000/api/nodes/4 | findstr /r "^[ ]*\"target\": 4$"
```

3. So, for example, I currently have a scenario where there are two connections to Node 4 from other nodes. But only one of them has a sequence to ID 4. In my case, that's Node 17.

```
curl http://localhost:5000/api/nodes/4
```

From this listing, I have the entire list of connection IDs and nodes. So, I can delete connection between #17 and #4.

```
curl -X DELETE http://localhost:5000/api/connections/18
```

Because I know from this listing that there are 2 connections and what interests me here is connection ID 18. I simply delete it and the node with ROOT #4 executes the sequence from #4 again

<h3>Real life scenario</h3>

Using Python scripts and request library. Let's say we want to create a chain of connections, 20 at a time. We can do this through a Python script. Let's assume we want to add this to ID 26, which already exists. But this naive approach only works if the IDs are actually are in numerical order. If they're heavily mixed up, it won't create a chain. Unless you go to admin panel and look at the last node on the list, meaning the last ID, you can start adding IDs from that point. But this is just an example.

```
# pip install requests
# python post_nodes.py
#
# Quickstart - how to get started with Requests - https://requests.readthedocs.io/en/latest/user/quickstart/
#

import requests

url = "http://localhost:5000/api/nodes"
headers = {"Content-Type": "application/json"}

start_value = 26           # change start ID node
loops = 20                 # number of nodes

for i in range(loops):
    connect_to = start_value + i
    payload = {
        "name": f"#{connect_to} My Node",
        "connect_to": connect_to,
        "sequence": "true"
    }
    response = requests.post(url, headers=headers, json=payload)
    print(f"Request {i+1}: connect_to={connect_to}, status={response.status_code}, response={response.text}")

```

<h3>Trick to quickly go to NODE you are looking for</h3>
In UI service running on a web browser, you'll see a little "search engine" on the left menu. Enter key in "Quick selection by ID" is not currently implemented. Now is the only way to click "Go" button after entering the ID in search box. However, if you know the node you're looking for, or you want to play and don't want to constantly zoom in and out, and scroll closer to node, you can simply enter, for example, #127 in box and press "GO" several times. Pressing it several times zooms in. Then, in the upper right corner, there's "FIT" to zoom out.


<br /><br />
There's no playback API. Like many other things, but you get the idea.

<h3>Update logs</h3>
28-08-2025 - Why am I adding the GPT 4o tokenizer to the exercises? The script itself isn't suitable for processing sequences like transfomer does, but it might be useful for writing scenarios using text fields. Perhaps, however, a 2D classifier for "Euclidean space" tokens could be created. For now, it's more for testing purposes to see if this approach makes sense. Apart from that, it is generally a script for writing more decision-making scenarios. Each node contains an X and Y location, a position on the canvas. Therefore, these graphs can be constructed in specific coordinate areas, either for the entire graph or for individual nodes. But for now, these are just ideas that need to be tested. This is a classification task, just a different type of data, graphs. But I don't want to get fixated on it either.

