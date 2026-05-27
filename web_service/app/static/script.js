const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, 320, 320);
ctx.fillStyle = 'black';

let sessionId = Date.now();

function getPixelMatrix() {
    const pixels = [];
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const pixel = ctx.getImageData(x*10, y*10, 1, 1).data;
            const brightness = (pixel[0] + pixel[1] + pixel[2]) / 3;
            pixels.push(brightness < 128 ? 255 : 0);
        }
    }
    return pixels;
}

function drawPixel(x, y) {
    const gridX = Math.floor(x / 10) * 10;
    const gridY = Math.floor(y / 10) * 10;
    ctx.fillRect(gridX, gridY, 10, 10);
}

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    drawPixel(e.offsetX, e.offsetY);
});
canvas.addEventListener('mousemove', (e) => {
    if (drawing) drawPixel(e.offsetX, e.offsetY);
});
canvas.addEventListener('mouseup', () => drawing = false);

canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.touches[0].clientX - rect.left) * scaleX;
    const y = (e.touches[0].clientY - rect.top) * scaleY;
    drawing = true;
    drawPixel(x, y);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.touches[0].clientX - rect.left) * scaleX;
    const y = (e.touches[0].clientY - rect.top) * scaleY;
    drawPixel(x, y);
});
canvas.addEventListener('touchend', () => drawing = false);

document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 320, 320);
    ctx.fillStyle = 'black';
    document.getElementById('result').innerHTML = '';
    document.getElementById('probabilities').innerHTML = '';
});

document.getElementById('recognizeBtn').addEventListener('click', async () => {
    const pixels = getPixelMatrix();
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels: pixels, session_id: sessionId })
    });
    const data = await response.json();
    
    document.getElementById('result').innerHTML = `
        Результат: <span style="color:#4CAF50">${data.digit}</span><br>
        Уверенность: ${(data.confidence * 100).toFixed(1)}%
    `;
    
    let probsHtml = '<h3>Вероятности:</h3>';
    data.probabilities.forEach((prob, idx) => {
        probsHtml += `<div>${idx}: ${(prob * 100).toFixed(1)}%
                      <div class="prob-bar" style="width: ${prob * 100}%"></div></div>`;
    });
    document.getElementById('probabilities').innerHTML = probsHtml;
    
    const statsRes = await fetch(`/api/stats/${sessionId}`);
    const stats = await statsRes.json();
    document.getElementById('stats').innerHTML = `
        Всего распознаваний в сессии: ${stats.total}<br>
        Средняя уверенность: ${(stats.avg_confidence * 100).toFixed(1)}%
    `;
});