pipeline {
  agent {
    kubernetes {
      // Jenkins Kubernetes Plugin 필요
      yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: kaniko-ci
spec:
  serviceAccountName: default
  containers:
  - name: kaniko
    image: gcr.io/kaniko-project/executor:latest
    command:
    - cat
    tty: true
    volumeMounts:
    - name: docker-config
      mountPath: /kaniko/.docker
  volumes:
  - name: docker-config
    emptyDir: {}
"""
    }
  }

  options {
    disableConcurrentBuilds()
    timestamps()
    ansiColor('xterm')
    buildDiscarder(logRotator(numToKeepStr: '20'))
    timeout(time: 30, unit: 'MINUTES')
  }

  environment {
    // ====== 환경값: 네 값으로 수정 ======
    HARBOR_REG = 'amdp-registry.skala-ai.com'     // Harbor 도메인
    HARBOR_NS  = 'skala25a'                        // Harbor 프로젝트(네임스페이스)
    FE_IMG     = "${HARBOR_REG}/${HARBOR_NS}/sk018-fe"
    BE_IMG     = "${HARBOR_REG}/${HARBOR_NS}/sk018-be"
    // Kaniko 캐시(선택): 없으면 자동 생성됨
    CACHE_REPO = "${HARBOR_REG}/${HARBOR_NS}/kaniko-cache"
    // Git 사용자 정보
    GIT_EMAIL  = 'lasnier@naver.com'
    GIT_NAME   = 'KEEHOON WON'
    // ====== 크리덴셜 ID ======
    HARBOR_DOCKERCFG = 'skala-image-registry-id'     // Secret file (config.json)
    GIT_TOKEN_ID     = 'skala-github-idtoken'                // String (PAT)

    // 공통
    SHORT_SHA = ''
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.SHORT_SHA = sh(script: "git rev-parse --short=12 HEAD", returnStdout: true).trim()
          echo "SHORT_SHA=${env.SHORT_SHA}"
        }
      }
    }

    stage('Build & Push Images (Kaniko, parallel)') {
      parallel {
        stage('FE') {
          steps {
            container('kaniko') {
              withCredentials([file(credentialsId: "${env.HARBOR_DOCKERCFG}", variable: 'DOCKER_CONFIG_JSON')]) {
                sh '''
                  set -eux
                  # Kaniko docker config 세팅
                  cp "$DOCKER_CONFIG_JSON" /kaniko/.docker/config.json

                  # 필수 파일 확인
                  test -f frontend/Dockerfile

                  /kaniko/executor \
                    --context="${PWD}" \
                    --dockerfile=frontend/Dockerfile \
                    --destination="${FE_IMG}:${SHORT_SHA}" \
                    --destination="${FE_IMG}:latest" \
                    --cache=true \
                    --cache-repo="${CACHE_REPO}" \
                    --use-new-run
                  echo "${SHORT_SHA}" > fe.sha
                '''
              }
            }
          }
        }
        stage('BE') {
          steps {
            container('kaniko') {
              withCredentials([file(credentialsId: "${env.HARBOR_DOCKERCFG}", variable: 'DOCKER_CONFIG_JSON')]) {
                sh '''
                  set -eux
                  cp "$DOCKER_CONFIG_JSON" /kaniko/.docker/config.json
                  test -f backend/Dockerfile

                  /kaniko/executor \
                    --context="${PWD}" \
                    --dockerfile=backend/Dockerfile \
                    --destination="${BE_IMG}:${SHORT_SHA}" \
                    --destination="${BE_IMG}:latest" \
                    --cache=true \
                    --cache-repo="${CACHE_REPO}" \
                    --use-new-run
                  echo "${SHORT_SHA}" > be.sha
                '''
              }
            }
          }
        }
      }
    }

    stage('Patch K8s Manifests & Commit') {
      steps {
        sh '''
          set -eux
          FE_SHA=$(cat fe.sha); BE_SHA=$(cat be.sha)

          # 매니페스트 태그 치환
          # image: <REG>/<NS>/sk018-fe:anything -> image: <REG>/<NS>/sk018-fe:${FE_SHA}
          sed -Ei "s#(image:[[:space:]]*${FE_IMG}:).*#\\1${FE_SHA}#g" k8s/frontend/deploy.yaml
          sed -Ei "s#(image:[[:space:]]*${BE_IMG}:).*#\\1${BE_SHA}#g" k8s/backend/deploy.yaml

          echo '--- FE deploy image line ---'
          grep -n 'image:' k8s/frontend/deploy.yaml || true
          echo '--- BE deploy image line ---'
          grep -n 'image:' k8s/backend/deploy.yaml || true

          git config --global --add safe.directory '*'
          git config user.email "${GIT_EMAIL}"
          git config user.name  "${GIT_NAME}"
          git add k8s/frontend/deploy.yaml k8s/backend/deploy.yaml || true

          if ! git diff --cached --quiet; then
            git commit -m "ci: deploy images FE=${FE_SHA} BE=${BE_SHA}"
          else
            echo "No manifest changes; skip commit."
          fi
        '''
        withCredentials([string(credentialsId: "${env.GIT_TOKEN_ID}", variable: 'GIT_TOKEN')]) {
          sh '''
            set -eux
            # origin URL에 토큰 삽입
            REPO_URL=$(git config --get remote.origin.url | sed "s#https://#https://oauth2:${GIT_TOKEN}@#")
            # 기본 브랜치(main/master) 자동 탐지
            DEFAULT_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | cut -d'/' -f2 || echo "main")
            git push "$REPO_URL" HEAD:${DEFAULT_BRANCH}
          '''
        }
      }
    }
  }

  post {
    success {
      echo "✅ Build & Push & Patch 성공: ${env.SHORT_SHA}"
    }
    failure {
      echo "❌ 실패. 콘솔 로그 확인 요망."
    }
    always {
      archiveArtifacts artifacts: '*.sha', onlyIfSuccessful: false
    }
  }
}
