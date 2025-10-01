pipeline {
  agent any

  environment {
    // ====== 사용자 수정 영역 ======
    GIT_URL        = 'https://github.com/nOOne-is-hier/payshield.git'
    GIT_BRANCH     = 'main'                     // 필요시 master
    GIT_ID         = 'skala-github-id'          // GitHub PAT (username/password or token)
    REG_URL        = 'amdp-registry.skala-ai.com' // Harbor 도메인
    REG_PROJECT    = 'skala25a'                 // Harbor 프로젝트
    FE_NAME        = 'fe-deploy'                // 프론트 이미지명
    BE_NAME        = 'be-deploy'                // 백엔드 이미지명
    DOCKER_CRED_ID = 'skala-image-registry-id'  // Harbor 계정(Username/Password)
    // ============================

    REG        = "${REG_URL}/${REG_PROJECT}"
    FE_IMG     = "${REG}/${FE_NAME}"
    BE_IMG     = "${REG}/${BE_NAME}"
    SHORT_SHA  = ''
    DEFAULT_BRANCH = ''
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

          // origin/HEAD → 기본 브랜치, 실패 시 GIT_BRANCH, 그래도 없으면 main
          def defBranch = sh(
            script: "sh -c 'git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | cut -d\\'/' -f2 || echo ${GIT_BRANCH}'",
            returnStdout: true
          ).trim()
          if (!defBranch) { defBranch = 'main' }
          writeFile file: 'DEFAULT_BRANCH', text: defBranch + "\n"

          echo "SHORT_SHA=${sha}, DEFAULT_BRANCH=${defBranch}"
          sh 'docker version || (echo "[!] docker cli not found" && exit 1)'
        }
      }
    }

    stage('Docker Login (Harbor)') {
      steps {
        withCredentials([
          usernamePassword(credentialsId: "${DOCKER_CRED_ID}", usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')
        ]) {
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
          # sha 파일이 "FE_SHA=xxxx" 같은 형식이어도 안전하게 헥사만 추출
          FE_SHA=$(tr -cd '[:xdigit:]' < fe.sha)
          BE_SHA=$(tr -cd '[:xdigit:]' < be.sha)

          # image: <REG>/<NAME>:<oldtag> 의 "태그"만 새 해시로 교체 (레지스트리/이름은 그대로 유지)
          sed -Ei "s#(image:[[:space:]]*${REG}/${FE_NAME})[^[:space:]]+#\\1:${FE_SHA}#" k8s/frontend/deploy.yaml
          sed -Ei "s#(image:[[:space:]]*${REG}/${BE_NAME})[^[:space:]]+#\\1:${BE_SHA}#" k8s/backend/deploy.yaml

          echo '--- FE deploy image ---'; grep -n '^[[:space:]]*image:' k8s/frontend/deploy.yaml || true
          echo '--- BE deploy image ---'; grep -n '^[[:space:]]*image:' k8s/backend/deploy.yaml || true
        '''
      }
    }

    // deployment yaml의 Git 커밋/푸시
    stage('Git Commit & Push (gitops)') {
      steps {
        script {
          // 1) sha를 먼저 환경변수로 고정 (파일 의존 제거)
          env.SHORT_SHA = sh(script: "cat SHORT_SHA 2>/dev/null || git rev-parse --short=12 HEAD", returnStdout: true).trim()
          env.FE_SHA = env.SHORT_SHA
          env.BE_SHA = env.SHORT_SHA

          env.GIT_REPO_PATH = env.GIT_URL.replaceFirst(/^https?:\/\//, '')
          echo "gitRepoPath: ${env.GIT_REPO_PATH}"
        }

        sh '''
          set -eux
          git config --global --add safe.directory '*'
          git config --global user.name "skala-gitops"
          git config --global user.email "skala@skala-ai.com"

          # 2) 원격 최신화
          git fetch --all --prune || true

          # 3) 워킹트리 정리 후 gitops 체크아웃
          git reset --hard
          git clean -fd
          if git show-ref --verify --quiet refs/remotes/origin/gitops; then
            git checkout -B gitops origin/gitops
          elif git show-ref --verify --quiet refs/remotes/origin/main; then
            git checkout -B gitops origin/main
          else
            git checkout -B gitops
          fi

          # 4) image 태그만 SHORT_SHA로 교체 (레지스트리/이미지명 유지)
          sed -Ei "s#(image:[[:space:]]*${REG_URL}/${REG_PROJECT}/${FE_NAME})[^[:space:]]*#\\1:${FE_SHA}#" k8s/frontend/deploy.yaml || true
          sed -Ei "s#(image:[[:space:]]*${REG_URL}/${REG_PROJECT}/${BE_NAME})[^[:space:]]*#\\1:${BE_SHA}#" k8s/backend/deploy.yaml  || true

          echo '--- after patch ---'
          grep -n '^[[:space:]]*image:' k8s/frontend/deploy.yaml || true
          grep -n '^[[:space:]]*image:' k8s/backend/deploy.yaml  || true

          # 5) 커밋 준비
          git add -A k8s
          git status
        '''

        withCredentials([usernamePassword(
          credentialsId: "${env.GIT_ID}",
          usernameVariable: 'GIT_PUSH_USER',
          passwordVariable: 'GIT_PUSH_PASSWORD'
        )]) {
          sh '''
            set -eux
            if ! git diff --cached --quiet; then
              git commit -m "[AUTO] Update FE:${FE_SHA} BE:${BE_SHA}"
              git remote set-url origin "https://${GIT_PUSH_USER}:${GIT_PUSH_PASSWORD}@${GIT_REPO_PATH}"
              git push origin gitops --force-with-lease || git push origin gitops
              echo "Pushed to gitops"
            else
              echo "No changes to commit"
            fi
          '''
        }
      }
    }
  }

  post {
    success {
      echo "✅ FE/BE push & manifests updated: ${env.SHORT_SHA}"
    }
    failure {
      echo "❌ Pipeline failed. Check logs."
    }
    always {
      script {
        try {
          archiveArtifacts artifacts: '*.sha', onlyIfSuccessful: false
        } catch (e) {
          echo "skip archive: ${e.class.simpleName}"
        }
      }
    }
  }
}
