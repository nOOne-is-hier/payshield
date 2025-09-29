document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // 기본 폼 제출 방지

    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert('업로드할 파일을 선택해주세요.');
        return;
    }

    // CSV 파일인지 확인
    if (file.type !== 'text/csv') {
        alert('유효한 CSV 파일을 선택해주세요.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file); // 파일을 FormData에 추가

    // 업로드 요청 알림
    alert('업로드가 요청되었습니다.');

    // FormData를 사용하여 API로 전송
    fetch('http://localhost:8001/upload', { // http://3.37.157.56:8000/upload/
        method: 'POST',
        body: formData // FormData를 직접 전송
    })
    .then(response => response.json())
    .then(data => {
        console.log('success:', data);
        alert('업로드가 정상적으로 진행되었습니다.'); // 성공 응답 시 알림
        onUploadComplete(); // 업로드 완료 시 호출
    })
    .catch((error) => {
        console.error('error:', error);
        alert('업로드 중 오류가 발생했습니다.'); // 오류 발생 시 알림
    });
});

// 업로드가 완료된 후 호출될 함수
function onUploadComplete() {
    document.getElementById('fileName').style.display = 'block';
}
