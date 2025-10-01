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
          FE_SHA=$(cat fe.sha); BE_SHA=$(cat be.sha)

          # k8s 이미지 태그 치환 (frontend/backend 각각의 deploy.yaml)
          sed -Ei "s#(^[[:space:]]*image:[[:space:]]*).*$#\1${FE_IMG}:${FE_SHA}#g" k8s/frontend/deploy.yaml
          sed -Ei "s#(^[[:space:]]*image:[[:space:]]*).*$#\1${BE_IMG}:${BE_SHA}#g" k8s/backend/deploy.yaml

          echo '--- FE deploy image ---'
          grep -n 'image:' k8s/frontend/deploy.yaml || true
          echo '--- BE deploy image ---'
          grep -n 'image:' k8s/backend/deploy.yaml || true
        '''
      }
    }

      // deployment yaml의 Git 커밋/푸시
      stage('Git Commit & Push (gitops)') {
        steps {
          script {
            env.GIT_REPO_PATH = env.GIT_URL.replaceFirst(/^https?:\/\//, '')
            echo "gitRepoPath: ${env.GIT_REPO_PATH}"
          }

          sh '''
            set -eux
            git config --global --add safe.directory '*'
            git config --global user.name "skala-gitops"
            git config --global user.email "skala@skala-ai.com"

            git fetch origin || true

            # gitops 브랜치 체크아웃
            if git show-ref --verify --quiet refs/heads/gitops; then
              git checkout -f gitops
            elif git show-ref --verify --quiet refs/remotes/origin/gitops; then
              git checkout -B gitops origin/gitops
            else
              git checkout -B gitops origin/main || git checkout -B gitops main
            fi

            # (선택) namespace 삭제 반영
            [ -f k8s/namespace.yaml ] && git rm k8s/namespace.yaml || true

            # FE/BE 해시 로드
            FE_SHA=$(cat fe.sha)
            BE_SHA=$(cat be.sha)

            echo "FE_IMG=${FE_IMG}, FE_SHA=${FE_SHA}"
            echo "BE_IMG=${BE_IMG}, BE_SHA=${BE_SHA}"

            # 이름이 무엇이든 'image:' 라인 전체를 덮어씀 (generic)
            sed -Ei "s#(^[[:space:]]*image:[[:space:]]*).*$#\1${FE_IMG}:${FE_SHA}#g" k8s/frontend/deploy.yaml
            sed -Ei "s#(^[[:space:]]*image:[[:space:]]*).*$#\1${BE_IMG}:${BE_SHA}#g" k8s/backend/deploy.yaml

            echo '--- after patch (gitops) ---'
            grep -n 'image:' k8s/frontend/deploy.yaml || true
            grep -n 'image:' k8s/backend/deploy.yaml || true

            # 충돌 마커가 있으면 실패(깨진 YAML 방지)
            ! grep -R --line-number -E '^(<{7}|={7}|>{7})' k8s || { echo "Conflict markers found"; git --no-pager grep -nE '^(<{7}|={7}|>{7})' k8s; exit 2; }

            # 삭제/수정 모두 스테이징
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
                FE_SHA=$(cat fe.sha); BE_SHA=$(cat be.sha)
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
