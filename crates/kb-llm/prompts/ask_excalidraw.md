# Excalidraw-diagram ask

You are answering the user's question by producing one or more Excalidraw
diagrams alongside a brief markdown caption. The diagrams will be opened in
Obsidian with the Excalidraw plugin installed.

## Question

{{query}}

## Retrieved sources

{{sources}}

## Citation Manifest

The following citation keys map to specific source locations. Use these keys
when you need to cite a source in the caption.

{{citation_manifest}}

## Instructions

1. Decide what the question is best answered with: a single diagram, or a
   small set of related diagrams (typically 1–3). Pick the smallest set that
   genuinely helps. Do not invent structure that is not in the sources.
2. For each diagram, write **one** `.excalidraw` JSON file into this exact
   directory:

   `{{output_path}}`

   Use a kebab-case filename ending in `.excalidraw` (e.g.
   `system-boundaries.excalidraw`, `data-flow.excalidraw`). Do not write
   anything outside that directory and do not write any other extensions.
3. Each `.excalidraw` file must be a valid Excalidraw document at the top
   level. The minimum schema looks like this:

   ```json
   {
     "type": "excalidraw",
     "version": 2,
     "source": "kb",
     "elements": [ /* shape objects, see below */ ],
     "appState": { "viewBackgroundColor": "#ffffff", "gridSize": null },
     "files": {}
   }
   ```

   Every shape object in `elements` must include these fields, in addition to
   any type-specific ones:

   ```json
   {
     "id": "<unique-string>",
     "type": "rectangle | ellipse | diamond | arrow | line | text",
     "x": 0, "y": 0, "width": 100, "height": 60,
     "angle": 0,
     "strokeColor": "#1e1e1e",
     "backgroundColor": "transparent",
     "fillStyle": "solid",
     "strokeWidth": 2,
     "strokeStyle": "solid",
     "roughness": 1,
     "opacity": 100,
     "groupIds": [],
     "frameId": null,
     "boundElements": [],
     "updated": 1,
     "link": null,
     "locked": false,
     "seed": 1,
     "version": 1,
     "versionNonce": 1,
     "isDeleted": false
   }
   ```

   - For `text` shapes, also include `text`, `fontSize` (16, 20, 28, 36),
     `fontFamily` (1=Virgil, 2=Helvetica, 3=Cascadia), `textAlign`
     ("left"|"center"|"right"), `verticalAlign` ("top"|"middle"|"bottom"),
     `baseline`, and a `containerId` of `null` unless the text is bound to a
     parent shape.
   - For `arrow`/`line`, include `points` (array of `[x,y]` offsets,
     starting `[0,0]`), `startBinding`, `endBinding`, `lastCommittedPoint`,
     `startArrowhead`, `endArrowhead`. Use `null` where you have no value.
   - Lay shapes out so they don't overlap. Keep the canvas ≤ ~1200 wide.
4. After all files are written, reply with a short markdown caption (no code
   fences) that:
   - Explains in 2–4 sentences what each diagram shows.
   - Embeds each diagram with an Obsidian wikilink, e.g.
     `![[system-boundaries.excalidraw]]`.
   - Cites sources using `[N]` keys from the manifest where appropriate.
5. Do not embed any diagram inline in the reply text; only embed via
   `![[…]]` references to the files you wrote.
6. If the sources do not contain enough structural detail to draw a faithful
   diagram, do **not** invent one. Reply with a single line beginning
   `ERROR:` explaining what is missing, and do not write any files.
