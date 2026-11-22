# LaTeX Reader frontend

A single page for the formula reader: add images of formulas, convert them, and copy
the LaTeX. It calls the backend in `../backend`.

It is plain HTML, CSS and JavaScript. There is no build step and nothing to install.
The interface follows [Material Design 3](https://m3.material.io) and uses the
[Material Web](https://github.com/material-components/material-web) components (buttons,
list, tabs, select, progress), loaded from the jsDelivr CDN.

## Run

1. Start the backend (see `../backend/README.md`):

   ```
   cd backend
   C:/Users/krichi/lab-space-win/py3.12/Scripts/python.exe app.py
   ```

2. Open `frontend/index.html` in a browser. Double-clicking the file works. You can
   also serve the folder:

   ```
   cd frontend
   python -m http.server 8000      # then open http://localhost:8000
   ```

The badge in the top right turns green and names the model once the backend answers.
Until then it says "Server offline" and checks again every 5 seconds.

The page loads Material Web, KaTeX and its fonts (Google Fonts) from CDNs, so it needs an
internet connection. Without one, the controls show as unstyled text, the preview does
not render, and the page falls back to system fonts.

To run the page and the backend in Docker, use `docker compose up --build` in `kurs/` and
open `http://localhost:8080` (see "Docker" in `../backend/README.md`). In that image nginx
serves the page, `config.docker.js` replaces `config.js`, and `/api` goes to the backend
container.

### Backend address

The default is `http://127.0.0.1:5000`, set in `config.js`. You can override it for a
single visit with a query parameter: `index.html?api=http://192.168.0.10:5000`.

The backend allows every origin by default (`LATEX_OCR_CORS_ORIGINS=*`). If you limit
it, add the origin the page is served from, for example `http://localhost:8000`. A page
opened from disk sends the origin `null`, so it only works with `*`.

## Using the page

**Images panel**

- While images convert, a progress bar under the panel title shows how far the run is.
- Add images by dropping them on the dashed area, by clicking it to open the file
  picker, or by pasting a screenshot with Ctrl+V. Files that are not images, and files
  already in the list, are skipped.
- Each image shows its thumbnail, file name, size and status. After conversion it also
  shows the confidence and the LaTeX it read. The × button removes an image.
- **Convert** (or Ctrl+Enter) sends the images to `POST /api/predict` one at a time,
  so each result appears as soon as it is ready. If some images failed, the button
  becomes "Convert N more" and retries only those.
- **Clear** removes every image and empties the result. During a conversion it reads
  "Stop & clear" and cancels the requests still waiting.

Images get a warning badge when the result is probably wrong: confidence below 75%,
the model never finished the formula, or the braces do not pair up. These follow the
`confidence`, `complete` and `balanced` fields described in the backend README.

**Result panel**

- **Text** shows the converted text, one formula per image, separated by blank lines.
  The text box can be edited.
- **Preview** renders the text with KaTeX. A formula that cannot be rendered shows its
  source and the KaTeX error.
- The format menu switches between **LaTeX** (bare formulas) and **Markdown**
  (`$$ … $$` blocks).
- **Copy** copies the text to the clipboard.

The preview always renders what is in the text box, so your edits appear there.
Changing the format, or adding or removing results, rebuilds the text from the model's
output and **discards manual edits**.

The moon/sun button in the top bar switches between the light and dark color schemes.
Until you use it, the page follows the system setting.

The chosen view, format and color scheme are remembered in the browser's local storage.

## Files

| File | |
|---|---|
| `index.html` | page structure |
| `styles.css` | M3 color, type and shape tokens, dark mode, and the narrow-screen layout |
| `app.js` | uploads, requests to the backend, rendering, copying |
| `config.js` | backend address |
| `Dockerfile`, `nginx.conf`, `config.docker.js` | the nginx image used by `../compose.yaml` |

`app.js` is a classic script, not an ES module, because Chrome and Edge block module
scripts on pages opened from `file://`.
