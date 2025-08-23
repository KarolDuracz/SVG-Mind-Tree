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

<h3>These are sample queries and now the version for WINDOWS -> CMD and Powershell</h3>

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

- update ( edit ) description field to change the playback direction for a node ID 13 to 17

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

1. Scenario if you want to change direction but you don't have a connection created yet (i.e. this sets the connection between #14 and #4)

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

<br /><br />
There's no playback API. Like many other things, but you get the idea.
