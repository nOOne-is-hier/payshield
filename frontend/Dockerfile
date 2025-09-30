# 1) Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build


# 2) Runtime stage (nginx)
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# SPA 라우팅 404 → index.html 처리(필요 시)
RUN printf 'try_files $uri /index.html;\n' > /etc/nginx/conf.d/spa.conf
EXPOSE 80
CMD ["nginx","-g","daemon off;"]