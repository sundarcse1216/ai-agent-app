# Multi-stage: Node build -> static serve via nginx (also proxies /api/*
# to the backend service, so the browser only ever talks to one origin —
# same relative-path approach as the Vite dev proxy).

FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
