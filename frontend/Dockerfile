# syntax=docker/dockerfile:1
# LaTeX Reader frontend: the static page on nginx, which also forwards /api to the backend.
#   docker build -t latex-reader-frontend frontend

FROM nginx:1.28-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js /usr/share/nginx/html/
# The page calls the API on its own origin, and nginx passes it on.
COPY config.docker.js /usr/share/nginx/html/config.js

EXPOSE 80
