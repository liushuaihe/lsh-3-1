const CIFAR10_CLASSES_LOCAL = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
];

const AppState = {
    originalImage: null,
    heatmapData: null,
    currentThreshold: 0.5,
    viewMode: 'overlay',
    classificationResults: null,
    classIndices: null,
    allProbabilities: null,
    selectedClassIndex: null,
    isProcessing: false,
    performance: {
        inferenceMs: null,
        heatmapMs: null,
        totalMs: null
    },
    dataset: {
        loaded: false,
        info: null,
        currentPage: 1,
        perPage: 50,
        labelFilter: null,
        totalPages: 1,
        totalImages: 0
    }
};

const elements = {};

document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initEventListeners();
    loadSampleImages();
    initTabs();
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

    elements.probabilityChart = document.getElementById('probabilityChart');
    elements.chartBars = document.getElementById('chartBars');
    elements.classSwitchHint = document.getElementById('classSwitchHint');

    elements.perfMonitor = document.getElementById('perfMonitor');
    elements.perfInference = document.getElementById('perfInference');
    elements.perfHeatmap = document.getElementById('perfHeatmap');
    elements.perfTotal = document.getElementById('perfTotal');

    elements.inputTabs = document.querySelectorAll('.input-tab');
    elements.tabContents = document.querySelectorAll('.tab-content');
    elements.vizTabs = document.querySelectorAll('.viz-tab');
    elements.vizContents = document.querySelectorAll('.viz-content');

    elements.datasetUploadArea = document.getElementById('datasetUploadArea');
    elements.datasetFileInput = document.getElementById('datasetFileInput');
    elements.datasetUploadBtn = document.getElementById('datasetUploadBtn');
    elements.datasetInfo = document.getElementById('datasetInfo');
    elements.datasetStats = document.getElementById('datasetStats');
    elements.browseDatasetBtn = document.getElementById('browseDatasetBtn');
    elements.clearDatasetBtn = document.getElementById('clearDatasetBtn');

    elements.browserGrid = document.getElementById('browserGrid');
    elements.browserInfo = document.getElementById('browserInfo');
    elements.browserPagination = document.getElementById('browserPagination');
    elements.prevPageBtn = document.getElementById('prevPageBtn');
    elements.nextPageBtn = document.getElementById('nextPageBtn');
    elements.pageInfo = document.getElementById('pageInfo');
    elements.classFilter = document.getElementById('classFilter');
    elements.perPageSelect = document.getElementById('perPageSelect');
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
    
    elements.datasetUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.datasetFileInput.click();
    });
    elements.datasetUploadArea.addEventListener('click', () => elements.datasetFileInput.click());
    elements.datasetFileInput.addEventListener('change', handleDatasetFileSelect);
    
    elements.datasetUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.datasetUploadArea.classList.add('dragover');
    });
    elements.datasetUploadArea.addEventListener('dragleave', () => {
        elements.datasetUploadArea.classList.remove('dragover');
    });
    elements.datasetUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.datasetUploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadDataset(files[0]);
        }
    });
    
    elements.browseDatasetBtn.addEventListener('click', () => {
        switchVizTab('dataset');
        loadDatasetPage(1);
    });
    
    elements.clearDatasetBtn.addEventListener('click', handleClearDataset);
    
    elements.prevPageBtn.addEventListener('click', () => {
        if (AppState.dataset.currentPage > 1) {
            loadDatasetPage(AppState.dataset.currentPage - 1);
        }
    });
    
    elements.nextPageBtn.addEventListener('click', () => {
        if (AppState.dataset.currentPage < AppState.dataset.totalPages) {
            loadDatasetPage(AppState.dataset.currentPage + 1);
        }
    });
    
    elements.classFilter.addEventListener('change', () => {
        AppState.dataset.labelFilter = elements.classFilter.value || null;
        loadDatasetPage(1);
    });
    
    elements.perPageSelect.addEventListener('change', () => {
        AppState.dataset.perPage = parseInt(elements.perPageSelect.value);
        loadDatasetPage(1);
    });
}

function initTabs() {
    elements.inputTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            switchInputTab(tabName);
        });
    });
    
    elements.vizTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const vizName = tab.dataset.viz;
            switchVizTab(vizName);
        });
    });
}

function switchInputTab(tabName) {
    elements.inputTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === 'tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1));
    });
}

function switchVizTab(vizName) {
    elements.vizTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.viz === vizName);
    });
    elements.vizContents.forEach(content => {
        content.classList.toggle('active', content.id === 'viz' + vizName.charAt(0).toUpperCase() + vizName.slice(1));
    });
    
    if (vizName === 'dataset' && AppState.dataset.loaded) {
        loadDatasetPage(AppState.dataset.currentPage);
    }
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
        AppState.classIndices = data.class_indices;
        AppState.allProbabilities = data.all_probabilities;
        AppState.selectedClassIndex = data.class_indices ? data.class_indices[0] : null;

        updateResults(data.classes, data.probabilities, data.class_indices);
        renderProbabilityChart(data.all_probabilities, data.class_indices);
        updatePerformanceMonitor(data.inference_time_ms, data.heatmap_time_ms, data.total_time_ms);

        await renderImage(data.original_image, data.heatmap);

        updateInfo(data.inference_time_ms, file.name);

        elements.placeholderState.classList.add('hidden');
        elements.imageDisplay.classList.remove('hidden');
        elements.imageToolbar.classList.remove('hidden');
        elements.classSwitchHint.style.display = 'inline-flex';

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

function updateResults(classes, probabilities, classIndices) {
    elements.resultsContainer.innerHTML = '';

    classes.forEach((cls, index) => {
        const prob = probabilities[index];
        const classIdx = classIndices ? classIndices[index] : null;
        const isSelected = classIdx !== null && classIdx === AppState.selectedClassIndex;
        const item = document.createElement('div');
        item.className = `result-item clickable ${index === 0 ? 'top-1' : ''} ${isSelected ? 'active-class' : ''}`;
        item.dataset.classIndex = classIdx;
        item.innerHTML = `
            <div class="result-rank">${index + 1}</div>
            <div class="result-class">${cls}</div>
            <div class="result-bar">
                <div class="result-bar-fill" style="width: 0%"></div>
            </div>
            <div class="result-prob">${prob.toFixed(1)}%</div>
            <div class="result-action" title="查看该类别的热力图">
                <span class="action-icon">🔥</span>
            </div>
        `;
        item.addEventListener('click', () => {
            if (classIdx !== null) {
                switchClassHeatmap(classIdx);
            }
        });
        elements.resultsContainer.appendChild(item);

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const barFill = item.querySelector('.result-bar-fill');
                if (barFill) {
                    barFill.style.width = prob + '%';
                }
            });
        });
    });
}

function renderProbabilityChart(allProbabilities, topClassIndices) {
    if (!allProbabilities || allProbabilities.length === 0) {
        elements.probabilityChart.style.display = 'none';
        return;
    }

    elements.probabilityChart.style.display = 'block';
    elements.chartBars.innerHTML = '';

    const topIndices = new Set(topClassIndices || []);
    const maxProb = Math.max(...allProbabilities);

    allProbabilities.forEach((prob, idx) => {
        const bar = document.createElement('div');
        bar.className = `chart-bar ${topIndices.has(idx) ? 'is-top' : ''} ${idx === AppState.selectedClassIndex ? 'active' : ''}`;
        bar.dataset.classIndex = idx;

        const heightPercent = maxProb > 0 ? (prob / maxProb) * 100 : 0;

        const valueTooltip = prob >= 1 ? prob.toFixed(1) + '%' : prob.toFixed(2) + '%';

        bar.innerHTML = `
            <div class="chart-bar-value" style="height: ${Math.max(heightPercent, 2)}%" title="${valueTooltip}"></div>
            <div class="chart-bar-label">${CIFAR10_CLASSES_LOCAL[idx]}</div>
        `;

        bar.addEventListener('click', () => switchClassHeatmap(idx));
        elements.chartBars.appendChild(bar);
    });
}

function updatePerformanceMonitor(inferenceMs, heatmapMs, totalMs) {
    AppState.performance.inferenceMs = inferenceMs;
    AppState.performance.heatmapMs = heatmapMs;
    AppState.performance.totalMs = totalMs;

    if (inferenceMs !== undefined && inferenceMs !== null) {
        elements.perfInference.textContent = inferenceMs + ' ms';
    }
    if (heatmapMs !== undefined && heatmapMs !== null) {
        elements.perfHeatmap.textContent = heatmapMs + ' ms';
    }
    if (totalMs !== undefined && totalMs !== null) {
        elements.perfTotal.textContent = totalMs + ' ms';
    }

    elements.perfMonitor.classList.add('pulse');
    setTimeout(() => {
        elements.perfMonitor.classList.remove('pulse');
    }, 600);
}

async function switchClassHeatmap(classIndex) {
    if (!AppState.originalImage || classIndex === null || classIndex === undefined) {
        return;
    }

    if (AppState.selectedClassIndex === classIndex) {
        return;
    }

    try {
        showLoading(
            `切换至「${CIFAR10_CLASSES_LOCAL[classIndex]}」热力图`,
            '重新计算 Grad-CAM...'
        );

        const response = await fetch('/api/gradcam-for-class', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                original_image: AppState.originalImage,
                class_index: classIndex
            })
        });

        if (!response.ok) {
            throw new Error('服务器响应错误');
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error('热力图生成失败');
        }

        AppState.selectedClassIndex = classIndex;
        AppState.heatmapData = data.heatmap;

        updatePerformanceMonitor(
            AppState.performance.inferenceMs,
            data.heatmap_time_ms,
            (AppState.performance.inferenceMs || 0) + data.heatmap_time_ms
        );

        const resultItems = elements.resultsContainer.querySelectorAll('.result-item');
        resultItems.forEach(item => {
            const idx = parseInt(item.dataset.classIndex, 10);
            item.classList.toggle('active-class', idx === classIndex);
        });

        const chartBars = elements.chartBars.querySelectorAll('.chart-bar');
        chartBars.forEach(bar => {
            const idx = parseInt(bar.dataset.classIndex, 10);
            bar.classList.toggle('active', idx === classIndex);
        });

        drawOverlay();

    } catch (error) {
        showError('热力图切换失败：' + error.message);
        console.error('GradCAM switch error:', error);
    } finally {
        hideLoading();
    }
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

function handleDatasetFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadDataset(file);
    }
}

async function uploadDataset(file) {
    const validTypes = ['.tar.gz', '.tgz', '.zip', '.bin'];
    const fileName = file.name.toLowerCase();
    const isValid = validTypes.some(type => fileName.endsWith(type));
    
    if (!isValid) {
        showError('请上传有效的数据集文件（.tar.gz、.zip 或 .bin 格式）');
        return;
    }

    try {
        showLoading('正在解析数据集...', '请稍候，这可能需要几秒钟');
        
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/dataset/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '数据集上传失败');
        }

        const data = await response.json();
        
        if (!data.success) {
            throw new Error('数据集解析失败');
        }

        AppState.dataset.loaded = true;
        AppState.dataset.info = data.dataset;
        AppState.dataset.totalImages = data.dataset.total_images;
        AppState.dataset.currentPage = 1;
        
        updateDatasetInfo(data.dataset);
        updateClassFilter(data.dataset.label_names);
        
        elements.datasetInfo.classList.remove('hidden');
        
        showSuccess('数据集加载成功！');
        
    } catch (error) {
        showError('数据集上传失败：' + error.message);
        console.error('Dataset upload error:', error);
    } finally {
        hideLoading();
    }
}

function updateDatasetInfo(datasetInfo) {
    elements.datasetStats.innerHTML = `
        <div class="stat-row">
            <span class="stat-label">图片总数</span>
            <span class="stat-value">${datasetInfo.total_images.toLocaleString()}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">类别数量</span>
            <span class="stat-value">${datasetInfo.num_classes}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">数据格式</span>
            <span class="stat-value">${datasetInfo.format}</span>
        </div>
    `;
}

function updateClassFilter(labelNames) {
    elements.classFilter.innerHTML = '<option value="">全部类别</option>';
    labelNames.forEach((name, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = name;
        elements.classFilter.appendChild(option);
    });
}

async function loadDatasetPage(page) {
    if (!AppState.dataset.loaded) {
        return;
    }

    try {
        const params = new URLSearchParams({
            page: page,
            per_page: AppState.dataset.perPage
        });
        
        if (AppState.dataset.labelFilter !== null) {
            params.append('label', AppState.dataset.labelFilter);
        }

        const response = await fetch('/api/dataset/images?' + params.toString());
        
        if (!response.ok) {
            throw new Error('获取图片列表失败');
        }

        const data = await response.json();
        
        if (!data.success) {
            throw new Error('获取图片列表失败');
        }

        AppState.dataset.currentPage = data.page;
        AppState.dataset.totalPages = data.total_pages;
        AppState.dataset.totalImages = data.total;
        
        renderDatasetImages(data.images);
        updatePagination();
        updateBrowserInfo(data.total);
        
    } catch (error) {
        showError('加载图片列表失败：' + error.message);
        console.error('Dataset images error:', error);
    }
}

function renderDatasetImages(images) {
    if (images.length === 0) {
        elements.browserGrid.innerHTML = `
            <div class="grid-placeholder">
                <div class="placeholder-icon">🖼️</div>
                <div class="placeholder-title">暂无图片</div>
                <div class="placeholder-desc">该类别下没有图片</div>
            </div>
        `;
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'grid-images';

    images.forEach(imgData => {
        const item = document.createElement('div');
        item.className = 'grid-image-item';
        item.title = `#${imgData.index} - ${imgData.label_name}`;
        item.innerHTML = `
            <img src="data:image/png;base64,${imgData.image}" alt="${imgData.label_name}">
            <div class="grid-label">${imgData.label_name}</div>
        `;
        item.addEventListener('click', () => classifyDatasetImage(imgData.index));
        grid.appendChild(item);
    });

    elements.browserGrid.innerHTML = '';
    elements.browserGrid.appendChild(grid);
}

function updatePagination() {
    const { currentPage, totalPages } = AppState.dataset;
    
    elements.pageInfo.textContent = `第 ${currentPage} 页 / 共 ${totalPages} 页`;
    elements.prevPageBtn.disabled = currentPage <= 1;
    elements.nextPageBtn.disabled = currentPage >= totalPages;
    
    if (totalPages > 1) {
        elements.browserPagination.classList.remove('hidden');
    } else {
        elements.browserPagination.classList.add('hidden');
    }
}

function updateBrowserInfo(total) {
    const filterText = AppState.dataset.labelFilter !== null ? '（已筛选）' : '';
    elements.browserInfo.textContent = `共 ${total.toLocaleString()} 张图片${filterText}`;
}

async function classifyDatasetImage(index) {
    try {
        showLoading('正在分析图片...', '模型推理中');

        const response = await fetch(`/api/dataset/classify/${index}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('分类失败');
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error('分类失败');
        }

        AppState.classificationResults = data;
        AppState.heatmapData = data.heatmap;
        AppState.originalImage = data.original_image;
        AppState.classIndices = data.class_indices;
        AppState.allProbabilities = data.all_probabilities;
        AppState.selectedClassIndex = data.class_indices ? data.class_indices[0] : null;

        updateResults(data.classes, data.probabilities, data.class_indices);
        renderProbabilityChart(data.all_probabilities, data.class_indices);
        updatePerformanceMonitor(data.inference_time_ms, data.heatmap_time_ms, data.total_time_ms);

        await renderImage(data.original_image, data.heatmap);

        updateInfo(data.inference_time_ms, `#${index}`);

        elements.placeholderState.classList.add('hidden');
        elements.imageDisplay.classList.remove('hidden');
        elements.imageToolbar.classList.remove('hidden');
        elements.classSwitchHint.style.display = 'inline-flex';

        switchVizTab('visualize');

    } catch (error) {
        showError('图片分析失败：' + error.message);
        console.error('Dataset classify error:', error);
    } finally {
        hideLoading();
    }
}

async function handleClearDataset() {
    if (!confirm('确定要清除已加载的数据集吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/dataset/clear', {
            method: 'POST'
        });

        if (response.ok) {
            AppState.dataset.loaded = false;
            AppState.dataset.info = null;
            elements.datasetInfo.classList.add('hidden');
            elements.browserGrid.innerHTML = `
                <div class="grid-placeholder">
                    <div class="placeholder-icon">🖼️</div>
                    <div class="placeholder-title">暂无数据</div>
                    <div class="placeholder-desc">上传数据集后即可浏览图片</div>
                </div>
            `;
            elements.browserPagination.classList.add('hidden');
            elements.browserInfo.textContent = '请先上传数据集';
            showSuccess('数据集已清除');
        }
    } catch (error) {
        showError('清除数据集失败');
        console.error('Clear dataset error:', error);
    }
}

function showSuccess(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 32px;
        background: #10b981;
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        font-size: 14px;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(0)';
    });
    
    setTimeout(() => {
        toast.style.transform = 'translateX(400px)';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}
