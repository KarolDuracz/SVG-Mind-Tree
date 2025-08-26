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

3. Add some nodes in sequence and click "analyze path" to change direction to new target node and play.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/version_3/images_ver3/3.png?raw=true)

