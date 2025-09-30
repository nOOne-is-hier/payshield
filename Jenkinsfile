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

    stage('Git Commit & Push (gitops)') {
      steps {
        sh '''
          set -eux

          SHORT_SHA=$(cat SHORT_SHA 2>/dev/null || git rev-parse --short=12 HEAD)
          DEFAULT_BRANCH=$(cat DEFAULT_BRANCH 2>/dev/null || echo main)

          git config --global --add safe.directory '*'
          git config user.name  "skala-gitops"
          git config user.email "skala@skala-ai.com"

          # 최신 정보 가져오기
          git fetch origin || true

          # 현재 워킹 브랜치(예: main)의 k8s 파일 백업
          mkdir -p .gitops-tmp
          cp -a k8s/frontend/deploy.yaml .gitops-tmp/fe.yaml
          cp -a k8s/backend/deploy.yaml  .gitops-tmp/be.yaml

          # gitops 브랜치로 체크아웃 (원격/로컬 있으면 사용, 없으면 기본브랜치에서 생성)
          if git show-ref --verify --quiet refs/heads/gitops; then
            git checkout -f gitops
          elif git show-ref --verify --quiet refs/remotes/origin/gitops; then
            git checkout -B gitops origin/gitops
          else
            git checkout -B gitops "${DEFAULT_BRANCH}"
          fi

          # 백업본 반영 (워킹 트리에 최신 파일로 덮어쓰기)
          cp -a .gitops-tmp/fe.yaml k8s/frontend/deploy.yaml
          cp -a .gitops-tmp/be.yaml k8s/backend/deploy.yaml
          rm -rf .gitops-tmp

          # 이미지 태그 치환 (FE/BE 각각)
          # FE_IMG/BE_IMG는 env에서 전달됨. 라인 전체 치환 방식으로 안전하게 교체.
          sed -Ei "s#(image:[[:space:]]*${FE_IMG}:).*#\\1${SHORT_SHA}#g" k8s/frontend/deploy.yaml
          sed -Ei "s#(image:[[:space:]]*${BE_IMG}:).*#\\1${SHORT_SHA}#g" k8s/backend/deploy.yaml

          echo '--- FE deploy image ---'; grep -n 'image:' k8s/frontend/deploy.yaml || true
          echo '--- BE deploy image ---'; grep -n 'image:' k8s/backend/deploy.yaml || true

          git add k8s/frontend/deploy.yaml k8s/backend/deploy.yaml || true
          if ! git diff --cached --quiet; then
            git commit -m "[AUTO] gitops: FE=${SHORT_SHA} BE=${SHORT_SHA}"
          else
            echo "No manifest changes."
          fi
        '''
        withCredentials([usernamePassword(
          credentialsId: "${GIT_ID}",
          usernameVariable: 'GIT_PUSH_USER',
          passwordVariable: 'GIT_PUSH_PASS'
        )]) {
          sh '''
            set -eux
            REPO_PATH=$(git config --get remote.origin.url | sed -E 's#^https?://##')
            git remote set-url origin "https://${GIT_PUSH_USER}:${GIT_PUSH_PASS}@${REPO_PATH}"
            # gitops 브랜치로 푸시 (생성/갱신 모두 대응)
            git push origin gitops:gitops --force-with-lease || git push origin gitops
            echo "Pushed to origin/gitops"
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
