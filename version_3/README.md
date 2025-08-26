<h2>Demo 3</h2>

In this demo I wanted to take some text and put it into a graph instead of manually building a mind map. And just build the sequence. No unnecessary frills for now. Just chop the text into words, tokens, and build this graph here.
<br /><br />

I used this ( from index.html )

```
clearSceneBtn.addEventListener('click', async () => {
  const ok = confirm('Delete all nodes and connections from the scene? This will clear the database. Are you sure?');
  if (!ok) return;
  const typed = prompt('Type DELETE to confirm clearing the entire database.');
  if (typed !== 'DELETE') { alert('Confirmation mismatch — aborted.'); return; }
  try {
    const resp = await fetch('/api/admin/clear', { method: 'POST' });
    if (resp.ok) {
      await loadData();
      selectNode(null);
      alert('Scene cleared.');
    } else {
      const txt = await resp.text();
      alert('Failed to clear scene: ' + txt);
    }
  } catch (err) {
    alert('Error clearing scene: ' + err);
  }
});
```

1.  As in the image, click the button on the left. A window will open. Paste your text. ONLY A SMALL PIECE BECAUSE THIS SYSTEM IS VERY SLOW. Mine took ~37 seconds to grind a small piece!

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_3/images_ver3/1.png?raw=true)

2. Click play.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_3/images_ver3/2.png?raw=true)

3. Add some nodes in sequence and click "analyze path" to change direction to new target node and play. If you want to return to the previous path, there's no need to search for the last token in node. In this case, simply click "analyze path" and enter next token after token where direction change was made, meaning somewhere in node further down the path, the path should be played. In this case, simply enter #676.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_3/images_ver3/3.png?raw=true)

<h2>Summary</h2>

I'll write a summary of this repo here. I probably won't add more, because it's a base. And by default, I need it for something else, these graphs. But now I've started thinking about the data, whether such maps make sense, etc. Because I've started measuring various things, like distance on a 2D plane and between layers of individual neurons in networks. And not just Euclidean ones. (...) I won't post anything more for now. I'll just leave the last image that started all the fun with these memory maps. I simply classified a small network into three layers ( yellow, red, blue dots ) and started measuring the distance between neurons in 2D ( on this canvas on the right ) and between layers in the network. So...
<br /><br />
On this image, I've highlighted a neuron from layer 2 (second layer), and these tables show the distances to the rest of the neurons in this layer. There are four of them. Yellow dots. (...) There's also a 2D Cosine proj. on the right. But it would be better to simply use euclidian here to see the distances around the point I've highlighted in yellow. But I had cosine set here.
<br /><br />
So the conclusion was that neurons can be transformed to a 2d plane in this way... the rest of story is this repo.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_3/images_ver3/last%20words%20for%20this%20demos.png?raw=true)
