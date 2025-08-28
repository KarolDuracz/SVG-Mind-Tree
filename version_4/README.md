> [!NOTE]  
> There's a small error in the last examples, 4 and 5. This requires fixing the last two files, which convert and save directly to the graph.db file on disk. It's related to the node's location on the canvas. These fields are currently missing and are NULL, so they're all in the same position. TODO.

<h2>A quick description of what's here</h2>

Exercises 1-3 are scripts that send queries to the server via request. <br /><br />
Exercises 4-5 write directly to the .db database on disk. This is much faster. There's no delay in sending subsequent queries. It simply converts directly to the JSON format used by graph.db and writes to the database on disk.

<hr>

A few scripts that help a bit in using this.

1. Go to version_3 folder. These are the examples I built based on this version of the code.
2. Run version 3 code > python app.py ( localhost:5000 )
3. Clear the database ( clear scene ) - it's better to use a clean canvas for these exercises

<h3>1. Exercise #1 - pre-processing before sending to the server using tiktoken.get_encoding("cl100k_base")</h3>

```
@file post5.py
```

POST canonical nodes (one per unique token id) — record token_id -> canonical_node_id.


POST instance nodes (one per position) — for each instance, if the token was repeated (or canonical exists), include connect_to canonical_node_id in the POST (so the server will create canonical -> instance connection automatically).


POST adjacency connections prev_instance_node_id -> current_instance_node_id (for sequence edges), deduplicated.

Script prints token stats (top-10 tokens) before sending

* **Planned first**: the whole graph (canonical nodes, instance nodes, adjacency edges) is planned in memory before any network calls (the `plan_graph` function).
* **Canonical nodes**: each unique tokenizer token id is represented by one canonical server node (created or reused). The token id is stored in the node's `description` field.
* **Instance nodes**: for each token occurrence we create an instance node. If the token is repeated (canonical exists), the instance is created with `connect_to` canonical\_node (so canonical → instance edge is automatically created by server).
* **Branching**: because repeated tokens connect instances to their canonical node, multiple instance nodes will attach to the same canonical node — visually producing branches on the canvas.
* **Sequence**: adjacency connections between consecutive positions are created later (POST /api/connections), preserving sequential order.
* **All sending happens after planning**: you confirm before sending; the script then posts canonical nodes first, instance nodes next (with connect\_to pointing to canonical), then adjacency edges.
 
Processing these three steps for this small input.txt file takes about 2 minutes in my case. Very slow. And very computationally demanding, especially with large files, because the graph is very large. This approach is very CPU-intensive. That's why I only put a small code x2 for example.

RUN 
```
pip install requests tiktoken
python post5.py --file input.txt --maxlen 6 --delay 0.02
# or
python post5.py --text "some code here" --maxlen 6
```

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_4/images_ver4/exercise1.png?raw=true)

<h3>2. Exercise #2 - Rescaling the position, i.e. expanding the graph in space on the canvas, e.g. x10 </h3>

```
@file scale_nodes_space.py
```

* fetches all nodes from `GET /api/nodes` and saves them to a local CSV,
* extracts each node's `id`, `x`, `y` (skips nodes missing coordinates, but logs them),
* computes the bounding box (minX/maxX/minY/maxY) and center,
* **scales each node position away from the bounding-box center** by a linear factor (default `10`) so the layout expands by that factor,
* writes the updated positions back to the server with `PUT /api/nodes/<id>` (one request per node),
* saves the updated node list to a second CSV and prints detailed logs.

Scaling choice: I scale linearly about the bounding-box center:

```
new_x = center_x + (x - center_x) * scale
new_y = center_y + (y - center_y) * scale
```

This multiplies linear distances from center by `scale` (default 10), so the bounding box grows by the same factor in each dimension.


RUN
```
pip install requests
python scale_nodes_space.py --url http://localhost:5000 --scale 10
# dry run
python scale_nodes_space.py --url http://localhost:5000 --scale 10 --dry-run
# specify output csv names and delay between PUTs
python scale_nodes_space.py --url http://localhost:5000 --scale 10 --out-before nodes_before.csv --out-after nodes_after.csv --delay 0.05
```

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_4/images_ver4/exercise2.png?raw=true)


<h3>3. Exercise #3 - processing the file to extract the patterns we are interested in </h3>

```
@file extract_and_build.py
```


Options:

* `--pattern` — regex pattern to match (default `if`).
* `--ignore-case` — case-insensitive matching.
* `--plain` — treat pattern as plain substring (not regex).
* `--out` — output file for matched lines (default `matched_lines.txt`).
* `--maxlen` — max substring length for recursive planning (default 6).
* `--delay` — delay between HTTP calls.
* `--dry-run` — do not send any network requests (still writes matched file and prints plan/stats).
* `--confirm` — skip the interactive confirmation and proceed (useful for automation).


RUN

```
python extract_and_build.py --input source.js --pattern "if (token === ' ')" --plain --out matches.txt
#
# or
#
python extract_and_build.py --input input.txt --pattern "if (token === ' ')" --plain --out matches.txt
#
# but in the case of the input.txt file from this folder with this piece of JS code it will only find e.g. "if "
#
python extract_and_build.py --input input.txt --pattern "if " --plain --out matches.txt 

```


<h3>4. Convert to JSON format to save directly to the database without using server queries. </h3>

```
@file build_plan_from_text.py
```

1. Reads `output.txt`.
2. Runs a GPT-4o tokenizer (via `tiktoken`) to get **tokens and token IDs**.
3. Builds a plan structure with:

   * `canonical_token_ids` = all unique token IDs.
   * `instances` = sequence of tokens (pos, token\_id, token\_text).
   * `adjacency_edges` = simple path edges from each token to the next (pos → pos+1).
4. Writes that as `plan.json`.

### Example run

```bash
python build_plan_from_text.py --input matches.txt --out plan.json --stats
```

Possible output:

```
[done] wrote plan with 528 instances, 361 canonical IDs to plan.json
[stats] unique token ids: 361
Top 10 tokens:
  id=284    text=' ='       count=36
  id=220    text=' '        count=34
  id=262    text='   '      count=32
  ...
```

json format for graph.db
```
{
  "canonical_token_ids": ["220", "284", "493", "738"],
  "instances": [
    {
      "pos": 0,
      "token_id": 284,
      "token_text": " =",
      "x": 42,
      "y": 73
    },
    {
      "pos": 1,
      "token_id": 220,
      "token_text": " ",
      "x": 52,
      "y": 83
    }
  ],
  "adjacency_edges": [[0,1],[1,2]]
}
```


<h3>5. Then you feed `plan.json` into `write_plan_to_db.py` to push directly into your `.db`.</h3>

```
@file write_plan_to_db.py
```


**Important:** make a backup copy of your `.db` file before running. Direct DB writes bypass server-side validation/layout code and cannot be undone by this script.

---

### Plan JSON format expected

```json
{
  "canonical_token_ids": ["284", "220", "738", ...],
  "instances": [
    {"pos": 0, "token_id": 284, "token_text": " if"},
    {"pos": 1, "token_id": 220, "token_text": " "},
    ...
  ],
  "adjacency_edges": [
    [0,1],
    [1,2],
    ...
  ]
}
```

`canonical_token_ids` are token-id strings you want a single canonical node created for (stored in node.description).
`instances` is a list in token-stream order with `pos` (integer), `token_id` (int), `token_text` (string).
`adjacency_edges` is list of `[src_pos, tgt_pos]` integer pairs for sequence adjacency.

---

### Usage examples

Dry-run (no DB changes):

```bash
python write_plan_to_db.py --db /path/to/your.db --plan plan.json --dry-run
```

Write into DB:

```bash
python write_plan_to_db.py --db /path/to/your.db --plan plan.json
```

Using graph.db from repo
```bash
python write_plan_to_db.py --db graph.db --plan plan.json
```

If your plan is in a different format, adapt it to the JSON layout above or ask me to modify the loader.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_4/images_ver4/exercise3.png?raw=true)
