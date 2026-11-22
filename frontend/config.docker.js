// config.js inside the Docker image (see Dockerfile). nginx serves the page and forwards
// /api to the backend container, so the backend lives on the page's own origin.
// A `?api=` query parameter still overrides it.
window.LATEX_OCR_API = location.origin;
