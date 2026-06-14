const AppState = {
    originalImage: null,
    heatmapData: null,
    currentThreshold: 0.5,
    viewMode: 'overlay',
    classificationResults: null,
    isProcessing: false
};

const elements = {};

document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initEventListeners();
    loadSampleImages();
});

function initElements() {
    elements.uploadArea = document.getElementById('uploadArea');
    elements.fileInput = document.getElementById('fileInput');
    elements.uploadBtn = document.getElementById('uploadBtn');
    elements.thresholdSlider = document.getElementById('thresholdSlider');
    elements.sliderValue = document.getElementById('sliderValue');
    elements.sliderFill = document.getElementById('sliderFill');
    elements.imageDisplay = document.getElementById('imageDisplay');
    elements.placeholderState = document.getElementById('placeholderState');
    elements.loadingOverlay = document.getElementById('loadingOverlay');
    elements.loadingText = document.getElementById('loadingText');
    elements.loadingSubtext = document.getElementById('loadingSubtext');
    elements.resultsContainer = document.getElementById('resultsContainer');
    elements.sampleGrid = document.getElementById('sampleGrid');
    elements.inferenceTime = document.getElementById('inferenceTime');
    elements.imageSize = document.getElementById('imageSize');
    elements.viewToggleOriginal = document.getElementById('viewToggleOriginal');
    elements.viewToggleOverlay = document.getElementById('viewToggleOverlay');
    elements.viewToggleHeatmap = document.getElementById('viewToggleHeatmap');
    elements.imageToolbar = document.getElementById('imageToolbar');
    elements.downloadBtn = document.getElementById('downloadBtn');
    elements.errorToast = document.getElementById('errorToast');
}

function initEventListeners() {
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.fileInput.click();
    });
    elements.fileInput.addEventListener('change', handleFileSelect);

    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    });

    elements.thresholdSlider.addEventListener('input', handleThresholdChange);

    elements.viewToggleOriginal.addEventListener('click', () => setViewMode('original'));
    elements.viewToggleOverlay.addEventListener('click', () => setViewMode('overlay'));
    elements.viewToggleHeatmap.addEventListener('click', () => setViewMode('heatmap'));

    elements.downloadBtn.addEventListener('click', handleDownload);
}

function loadSampleImages() {
    const sampleImages = [
        { name: '飞机', url: '/static/images/samples/airplane.png' },
        { name: '汽车', url: '/static/images/samples/automobile.png' },
        { name: '鸟', url: '/static/images/samples/bird.png' },
        { name: '猫', url: '/static/images/samples/cat.png' },
        { name: '鹿', url: '/static/images/samples/deer.png' },
        { name: '狗', url: '/static/images/samples/dog.png' },
        { name: '青蛙', url: '/static/images/samples/frog.png' },
        { name: '马', url: '/static/images/samples/horse.png' },
        { name: '船', url: '/static/images/samples/ship.png' },
        { name: '卡车', url: '/static/images/samples/truck.png' }
    ];

    sampleImages.forEach((sample, index) => {
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.title = sample.name;
        item.innerHTML = `<img src="${sample.url}" alt="${sample.name}" loading="lazy">`;
        item.addEventListener('click', () => loadSampleImage(sample.url));
        elements.sampleGrid.appendChild(item);
    });
}

async function loadSampleImage(url) {
    try {
        showLoading('正在加载示例图片...', '请稍候');
        
        const response = await fetch(url);
        const blob = await response.blob();
        const file = new File([blob], 'sample.jpg', { type: blob.type });
        
        await processFile(file);
    } catch (error) {
        showError('示例图片加载失败，请尝试上传本地图片');
        console.error('Sample image load error:', error);
    } finally {
        hideLoading();
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

async function processFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('请上传图片文件');
        return;
    }

    try {
        showLoading('正在分析图片...', '模型推理中');
        
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('服务器响应错误');
        }

        const data = await response.json();
        
        if (!data.success) {
            throw new Error('分类失败');
        }

        AppState.classificationResults = data;
        AppState.heatmapData = data.heatmap;
        AppState.originalImage = data.original_image;

        updateResults(data.classes, data.probabilities);
        
        await renderImage(data.original_image, data.heatmap);
        
        updateInfo(data.inference_time, file.name);
        
        elements.placeholderState.classList.add('hidden');
        elements.imageDisplay.classList.remove('hidden');
        elements.imageToolbar.classList.remove('hidden');
        
    } catch (error) {
        showError('图片处理失败：' + error.message);
        console.error('Classification error:', error);
    } finally {
        hideLoading();
    }
}

async function renderImage(originalBase64, heatmapData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            canvas.className = 'image-display';
            canvas.id = 'imageCanvas';
            
            elements.imageDisplay.innerHTML = '';
            elements.imageDisplay.appendChild(canvas);
            
            AppState.imageCanvas = canvas;
            AppState.imageWidth = img.width;
            AppState.imageHeight = img.height;
            AppState.originalImageData = img;
            
            drawOverlay();
            resolve();
        };
        img.src = 'data:image/png;base64,' + originalBase64;
    });
}

function drawOverlay() {
    if (!AppState.imageCanvas || !AppState.originalImageData || !AppState.heatmapData) {
        return;
    }

    const canvas = AppState.imageCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (AppState.viewMode === 'original') {
        ctx.drawImage(AppState.originalImageData, 0, 0);
        return;
    }

    ctx.drawImage(AppState.originalImageData, 0, 0);

    const heatmap = AppState.heatmapData;
    const heatmapRows = heatmap.length;
    const heatmapCols = heatmap[0].length;

    const threshold = AppState.currentThreshold;

    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = heatmapCols;
    heatmapCanvas.height = heatmapRows;
    const heatmapCtx = heatmapCanvas.getContext('2d');
    const imageData = heatmapCtx.createImageData(heatmapCols, heatmapRows);

    for (let y = 0; y < heatmapRows; y++) {
        for (let x = 0; x < heatmapCols; x++) {
            const value = heatmap[y][x];
            const idx = (y * heatmapCols + x) * 4;

            if (value >= threshold) {
                const normalizedValue = (value - threshold) / (1 - threshold);
                const color = jetColormap(normalizedValue);
                imageData.data[idx] = color.r;
                imageData.data[idx + 1] = color.g;
                imageData.data[idx + 2] = color.b;
                imageData.data[idx + 3] = Math.floor(255 * 0.6 * normalizedValue);
            } else {
                imageData.data[idx] = 0;
                imageData.data[idx + 1] = 0;
                imageData.data[idx + 2] = 0;
                imageData.data[idx + 3] = 0;
            }
        }
    }

    heatmapCtx.putImageData(imageData, 0, 0);

    if (AppState.viewMode === 'heatmap') {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(heatmapCanvas, 0, 0, width, height);
    } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(heatmapCanvas, 0, 0, width, height);
    }
}

function jetColormap(value) {
    value = Math.max(0, Math.min(1, value));
    
    let r, g, b;
    
    if (value < 0.125) {
        r = 0;
        g = 0;
        b = 0.5 + 4 * value;
    } else if (value < 0.375) {
        r = 0;
        g = 4 * (value - 0.125);
        b = 1;
    } else if (value < 0.625) {
        r = 4 * (value - 0.375);
        g = 1;
        b = 1 - 4 * (value - 0.375);
    } else if (value < 0.875) {
        r = 1;
        g = 1 - 4 * (value - 0.625);
        b = 0;
    } else {
        r = 1 - 4 * (value - 0.875);
        g = 0;
        b = 0;
    }
    
    return {
        r: Math.floor(r * 255),
        g: Math.floor(g * 255),
        b: Math.floor(b * 255)
    };
}

function handleThresholdChange(e) {
    const value = parseFloat(e.target.value);
    AppState.currentThreshold = value;
    
    elements.sliderValue.textContent = (value * 100).toFixed(0) + '%';
    elements.sliderFill.style.width = (value * 100) + '%';
    
    requestAnimationFrame(drawOverlay);
}

function setViewMode(mode) {
    AppState.viewMode = mode;
    
    elements.viewToggleOriginal.classList.toggle('active', mode === 'original');
    elements.viewToggleOverlay.classList.toggle('active', mode === 'overlay');
    elements.viewToggleHeatmap.classList.toggle('active', mode === 'heatmap');
    
    drawOverlay();
}

function updateResults(classes, probabilities) {
    elements.resultsContainer.innerHTML = '';
    
    classes.forEach((cls, index) => {
        const prob = probabilities[index];
        const item = document.createElement('div');
        item.className = `result-item ${index === 0 ? 'top-1' : ''}`;
        item.innerHTML = `
            <div class="result-rank">${index + 1}</div>
            <div class="result-class">${cls}</div>
            <div class="result-bar">
                <div class="result-bar-fill" style="width: ${prob}%"></div>
            </div>
            <div class="result-prob">${prob.toFixed(1)}%</div>
        `;
        elements.resultsContainer.appendChild(item);
    });
}

function updateInfo(inferenceTime, fileName) {
    elements.inferenceTime.textContent = inferenceTime + ' ms';
    
    if (AppState.imageWidth && AppState.imageHeight) {
        elements.imageSize.textContent = `${AppState.imageWidth} × ${AppState.imageHeight}`;
    }
}

function showLoading(text, subtext = '') {
    AppState.isProcessing = true;
    elements.loadingText.textContent = text;
    elements.loadingSubtext.textContent = subtext;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    AppState.isProcessing = false;
    elements.loadingOverlay.classList.remove('active');
}

function showError(message) {
    elements.errorToast.textContent = message;
    elements.errorToast.classList.add('show');
    
    setTimeout(() => {
        elements.errorToast.classList.remove('show');
    }, 3000);
}

function handleDownload() {
    if (!AppState.imageCanvas) return;
    
    const link = document.createElement('a');
    link.download = 'gradcam-result.png';
    link.href = AppState.imageCanvas.toDataURL('image/png');
    link.click();
}
