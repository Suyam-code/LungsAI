const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultDiv = document.getElementById('result');
const diagnosisText = document.getElementById('diagnosisText');
const confidenceText = document.getElementById('confidenceText');
let probChart = null;

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    } else {
        previewImg.style.display = 'none';
        analyzeBtn.disabled = true;
        resultDiv.style.display = 'none';
    }
});

analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.status === 'error') {
            diagnosisText.textContent = '⚠️ Error';
            confidenceText.textContent = data.message;
            resultDiv.style.display = 'block';
            if (probChart) probChart.destroy();
        } else {
            diagnosisText.textContent = `🩺 Diagnosis: ${data.diagnosis}`;
            confidenceText.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
            resultDiv.style.display = 'block';

            // Update chart
            const labels = Object.keys(data.probabilities);
            const probs = Object.values(data.probabilities);

            if (probChart) probChart.destroy();
            const ctx = document.getElementById('probChart').getContext('2d');
            probChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Probability',
                        data: probs,
                        backgroundColor: ['#6c757d', '#dc3545', '#28a745', '#ffc107'],
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: { beginAtZero: true, max: 1, title: { display: true, text: 'Probability' } }
                    }
                }
            });
        }
    } catch (err) {
        diagnosisText.textContent = '⚠️ Connection Error';
        confidenceText.textContent = 'Could not reach the server. Make sure the backend is running.';
        resultDiv.style.display = 'block';
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze X‑ray';
    }
});