pipeline {
  agent any

  environment {
    // === 바꿔써야 하는 값들 ===
    GIT_URL        = 'https://github.com/nOOne-is-hier/payshield.git'
    GIT_BRANCH     = 'main'                     // 소스 빌드용 브랜치
    GIT_ID         = 'skala-github-id'          // GitHub PAT (username/password or token)

    REG_URL        = 'amdp-registry.skala-ai.com' // Harbor 도메인
    REG_PROJECT    = 'skala25a'                 // Harbor 프로젝트
    DOCKER_CRED_ID = 'skala-image-registry-id'  // Harbor Credentials ID

    FE_NAME        = 'fe-deploy'                // 프론트 이미지명
    BE_NAME        = 'be-deploy'                // 백엔드 이미지명
    // =========================

    REG        = "${REG_URL}/${REG_PROJECT}"
    FE_IMG     = "${REG}/${FE_NAME}"
    BE_IMG     = "${REG}/${BE_NAME}"
  }

  options {
    disableConcurrentBuilds()
    timestamps()
    ansiColor('xterm')
  }

  stages {
    stage('Inject OPENAI Secret') {
      steps {
        withCredentials([string(credentialsId: 'openai-api-key-id', variable: 'OPENAI_API_KEY')]) {
          sh '''
            set -eu
            # 로그 마스킹 강화: -x 끄기
            set +x
            kubectl -n skala-practice create secret generic openai \
              --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
              --dry-run=client -o yaml | kubectl apply -f -
            # 다시 -x 켜도 됨
            set -x
          '''
        }
      }
    }

    stage('Checkout') {
      steps {
        git branch: "${GIT_BRANCH}", url: "${GIT_URL}", credentialsId: "${GIT_ID}"
        script {
          env.SHORT_SHA = sh(script: "git rev-parse --short=12 HEAD", returnStdout: true).trim()
          echo "SHORT_SHA=${env.SHORT_SHA}"
        }
      }
    }

    stage('Docker Login') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: "${DOCKER_CRED_ID}",
          usernameVariable: 'REG_USER',
          passwordVariable: 'REG_PASS'
        )]) {
          sh '''
            set -eux
            echo "$REG_PASS" | docker login "${REG_URL}" -u "$REG_USER" --password-stdin
          '''
        }
      }
    }

    stage('Build & Push FE') {
      steps {
        sh '''
          set -eux
          test -f frontend/Dockerfile
          docker build -t "${FE_IMG}:${SHORT_SHA}" -t "${FE_IMG}:latest" -f frontend/Dockerfile frontend
          docker push "${FE_IMG}:${SHORT_SHA}"
          docker push "${FE_IMG}:latest"
        '''
      }
    }

    stage('Build & Push BE') {
      steps {
        sh '''
          set -eux
          test -f backend/Dockerfile
          docker build -t "${BE_IMG}:${SHORT_SHA}" -t "${BE_IMG}:latest" -f backend/Dockerfile backend
          docker push "${BE_IMG}:${SHORT_SHA}"
          docker push "${BE_IMG}:latest"
        '''
      }
    }

    stage('Patch Manifests & Commit to gitops') {
      steps {
        // 깃 설정 및 gitops 브랜치 체크아웃
        sh '''
          set -eux
          git config --global --add safe.directory '*'
          git config --global user.name  "skala-gitops"
          git config --global user.email "skala@skala-ai.com"

          git fetch --all --prune || true
          # gitops 브랜치 준비 (원격 있으면 트래킹, 없으면 생성)
          if git show-ref --verify --quiet refs/remotes/origin/gitops; then
            git checkout -B gitops origin/gitops
          else
            git checkout -B gitops
          fi
        '''

        // 이미지 태그만 교체
        sh '''
          set -eux
          test -f k8s/frontend/deploy.yaml
          test -f k8s/backend/deploy.yaml

          echo "--- BEFORE (FE) ---"
          grep -n '^[[:space:]]*image:' k8s/frontend/deploy.yaml || true
          echo "--- BEFORE (BE) ---"
          grep -n '^[[:space:]]*image:' k8s/backend/deploy.yaml  || true

          # "image: REG/NAME:<anything>" 의 <anything>만 SHORT_SHA로 치환
          sed -Ei "s#(image:[[:space:]]*${REG}/${FE_NAME})[^[:space:]]*#\\1:${SHORT_SHA}#g" k8s/frontend/deploy.yaml
          sed -Ei "s#(image:[[:space:]]*${REG}/${BE_NAME})[^[:space:]]*#\\1:${SHORT_SHA}#g" k8s/backend/deploy.yaml

          echo "--- AFTER (FE) ---"
          grep -n '^[[:space:]]*image:' k8s/frontend/deploy.yaml || true
          echo "--- AFTER (BE) ---"
          grep -n '^[[:space:]]*image:' k8s/backend/deploy.yaml  || true

          git add k8s/frontend/deploy.yaml k8s/backend/deploy.yaml
        '''

        // 푸시
        withCredentials([usernamePassword(
          credentialsId: "${GIT_ID}",
          usernameVariable: 'GIT_PUSH_USER',
          passwordVariable: 'GIT_PUSH_PASSWORD'
        )]) {
          script {
            def repoPath = env.GIT_URL.replaceFirst(/^https?:\\/\\//, '')
            sh """
              set -eux
              if ! git diff --cached --quiet; then
                git commit -m "[AUTO] FE:${FE_NAME}:${SHORT_SHA} BE:${BE_NAME}:${SHORT_SHA}"
                git remote set-url origin "https://${GIT_PUSH_USER}:${GIT_PUSH_PASSWORD}@${repoPath}"
                git push origin gitops --force-with-lease || git push origin gitops
                echo "Pushed to gitops"
              else
                echo "No changes to commit"
              fi
            """
          }
        }
      }
    }
  }

  post {
    success {
      echo "✅ Done. Image tags updated to ${env.SHORT_SHA}"
    }
    failure {
      echo "❌ Failed. Check logs."
    }
  }
}
