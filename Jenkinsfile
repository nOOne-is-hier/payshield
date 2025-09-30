pipeline {
  agent any

  environment {
    // ====== 사용자 수정 영역 ======
    GIT_URL                = 'https://github.com/nOOne-is-hier/payshield.git'
    GIT_BRANCH             = 'main'                         // 필요시 master
    GIT_ID                 = 'skala-github-id'              // GitHub PAT (username/password or token)
    REG_URL                = 'amdp-registry.skala-ai.com'   // Harbor 도메인
    REG_PROJECT            = 'skala25a'                     // Harbor 프로젝트
    FE_NAME                = 'fe-deploy'                     // 프론트 이미지명
    BE_NAME                = 'be-deploy'                     // 백엔드 이미지명
    DOCKER_CRED_ID         = 'skala-image-registry-id'      // Harbor 계정(Username/Password)
    // ============================

    REG                    = "${REG_URL}/${REG_PROJECT}"
    FE_IMG                 = "${REG}/${FE_NAME}"
    BE_IMG                 = "${REG}/${BE_NAME}"
    SHORT_SHA              = ''
    DEFAULT_BRANCH         = ''
  }

  options {
    disableConcurrentBuilds()
    timestamps()
    ansiColor('xterm')
  }

  stages {
    stage('Checkout') {
      steps {
        git branch: "${GIT_BRANCH}", url: "${GIT_URL}", credentialsId: "${GIT_ID}"
        script {
          def sha = sh(script: "git rev-parse --short=12 HEAD", returnStdout: true).trim()
          writeFile file: 'SHORT_SHA', text: sha + "\n"
          // (옵션) 기본 브랜치: 실패해도 main으로
          def defBranch = sh(script: "sh -c 'git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | cut -d\\'/' -f2 || echo ${GIT_BRANCH}'", returnStdout: true).trim()
          writeFile file: 'DEFAULT_BRANCH', text: defBranch + "\n"
          echo "SHORT_SHA=${sha}, DEFAULT_BRANCH=${defBranch}"
          sh 'docker version || (echo "[!] docker cli not found" && exit 1)'
        }
      }
    }

    stage('Docker Login (Harbor)') {
      steps {
        withCredentials([usernamePassword(credentialsId: "${DOCKER_CRED_ID}", usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')]) {
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
          : "${SHORT_SHA:=$(cat SHORT_SHA 2>/dev/null || git rev-parse --short=12 HEAD)}"
          test -f frontend/Dockerfile

          docker build -t "${FE_IMG}:${SHORT_SHA}" -t "${FE_IMG}:latest" -f frontend/Dockerfile frontend
          docker push "${FE_IMG}:${SHORT_SHA}"
          docker push "${FE_IMG}:latest"
          echo "${SHORT_SHA}" > fe.sha
        '''
      }
    }

    stage('Build & Push BE') {
      steps {
        sh '''
          set -eux
          : "${SHORT_SHA:=$(cat SHORT_SHA 2>/dev/null || git rev-parse --short=12 HEAD)}"
          test -f backend/Dockerfile

          docker build -t "${BE_IMG}:${SHORT_SHA}" -t "${BE_IMG}:latest" -f backend/Dockerfile backend
          docker push "${BE_IMG}:${SHORT_SHA}"
          docker push "${BE_IMG}:latest"
          echo "${SHORT_SHA}" > be.sha
        '''
      }
    }

    stage('Patch Manifests (image tag update)') {
      steps {
        sh '''
          set -eux
          FE_SHA=$(cat fe.sha); BE_SHA=$(cat be.sha)

          # k8s 이미지 태그 치환 (frontend/backend 각각의 deploy.yaml)
          sed -Ei "s#(image:[[:space:]]*${FE_IMG}:).*#\\1${FE_SHA}#g" k8s/frontend/deploy.yaml
          sed -Ei "s#(image:[[:space:]]*${BE_IMG}:).*#\\1${BE_SHA}#g" k8s/backend/deploy.yaml

          echo '--- FE deploy image ---'
          grep -n 'image:' k8s/frontend/deploy.yaml || true
          echo '--- BE deploy image ---'
          grep -n 'image:' k8s/backend/deploy.yaml || true
        '''
      }
    }

    stage('Git Commit & Push (GitOps)') {
      steps {
        sh '''
          set -eux
          git config --global --add safe.directory '*'
          git config user.name  "skala-gitops"
          git config user.email "skala@skala-ai.com"
          git add k8s/frontend/deploy.yaml k8s/backend/deploy.yaml || true
          if ! git diff --cached --quiet; then
            git commit -m "ci: update images FE=${SHORT_SHA} BE=${SHORT_SHA}"
          else
            echo "No manifest changes."
          fi
        '''
        withCredentials([usernamePassword(credentialsId: "${GIT_ID}", usernameVariable: 'GIT_PUSH_USER', passwordVariable: 'GIT_PUSH_PASS')]) {
          sh '''
            set -eux
            REPO_PATH=$(git config --get remote.origin.url | sed -E 's#^https?://##')
            git remote set-url origin "https://${GIT_PUSH_USER}:${GIT_PUSH_PASS}@${REPO_PATH}"
            git push origin "HEAD:${DEFAULT_BRANCH}"
          '''
        }
      }
    }
  }

  post {
    success { echo "✅ FE/BE push & manifests updated: ${env.SHORT_SHA}" }
    failure { echo "❌ Pipeline failed. Check logs." }
    always {
      script {
        try { archiveArtifacts artifacts: '*.sha', onlyIfSuccessful: false } catch (e) { echo "skip archive: ${e.class.simpleName}" }
      }
    }
  }
}
