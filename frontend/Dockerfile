# 1) Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 2) Runtime stage (nginx)
FROM nginx:alpine
# 정적 파일 복사
COPY --from=build /app/dist /usr/share/nginx/html

# 기본 conf 지우고 우리 conf로 교체
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx","-g","daemon off;"]
