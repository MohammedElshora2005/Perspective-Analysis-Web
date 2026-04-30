let selectedFile = null;
let currentPerspective = '2';
let currentResultPath = '';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewCard = document.getElementById('previewCard');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const infoPanel = document.getElementById('infoPanel');
const downloadBtn = document.getElementById('downloadBtn');
const loadingOverlay = document.getElementById('loadingOverlay');

// Allowed image types
const ALLOWED_TYPES = ['image/png', 'image/jpg', 'image/jpeg', 'image/bmp', 'image/tiff'];
const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff'];

// Perspective selector buttons
document.querySelectorAll('.perspective-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.perspective-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentPerspective = btn.dataset.type;
        console.log(`Perspective changed to: ${currentPerspective}-point`);
    });
});

// Upload area click
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag & drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#a855f7';
    uploadArea.style.background = 'rgba(168,85,247,0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'rgba(255,255,255,0.2)';
    uploadArea.style.background = 'transparent';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(255,255,255,0.2)';
    uploadArea.style.background = 'transparent';
    
    const file = e.dataTransfer.files[0];
    if (file && validateFileType(file)) {
        handleFile(file);
    } else {
        showError(`Please drop an image file (${ALLOWED_EXTENSIONS.join(', ')})`);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        if (validateFileType(e.target.files[0])) {
            handleFile(e.target.files[0]);
        }
    }
});

function validateFileType(file) {
    // Check by MIME type
    if (!ALLOWED_TYPES.includes(file.type)) {
        showError(`File type not allowed. Please use: ${ALLOWED_EXTENSIONS.join(', ')}`);
        return false;
    }
    
    // Check by extension as backup
    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
        showError(`Extension '.${ext}' not allowed. Please use: ${ALLOWED_EXTENSIONS.join(', ')}`);
        return false;
    }
    
    return true;
}

function handleFile(file) {
    // Validate file size (max 32MB)
    if (file.size > 32 * 1024 * 1024) {
        showError('File is too large! Maximum size is 32MB.');
        return;
    }
    
    selectedFile = file;
    
    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewCard.style.display = 'block';
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
        analyzeBtn.style.cursor = 'pointer';
        
        // Scroll to preview
        previewCard.scrollIntoView({ behavior: 'smooth' });
    };
    reader.onerror = () => {
        showError('Error reading file. Please try again.');
    };
    reader.readAsDataURL(file);
    
    // Hide previous results
    resultsSection.style.display = 'none';
    currentResultPath = '';
    
    // Remove any existing error
    const oldError = document.querySelector('.error-message');
    if (oldError) oldError.remove();
    
    console.log(`File loaded: ${file.name}, Size: ${(file.size / 1024).toFixed(2)} KB, Type: ${file.type}`);
}

// Show error message in a nice way
function showError(message) {
    // Remove any existing error
    const oldError = document.querySelector('.error-message');
    if (oldError) oldError.remove();
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div style="background: rgba(255,0,0,0.2); border-left: 4px solid #ff4444; border-radius: 12px; padding: 15px; margin: 20px;">
            <span style="font-size: 20px;">❌</span>
            <strong style="color: #ff8888;">Error:</strong>
            <span style="color: white;"> ${message}</span>
        </div>
    `;
    
    // Add to results section or after upload card
    if (resultsSection) {
        resultsSection.insertBefore(errorDiv, resultsSection.firstChild);
        resultsSection.style.display = 'block';
    } else {
        document.querySelector('.main-content').after(errorDiv);
    }
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        if (errorDiv && errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 5000);
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Disable button while processing
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.5';
    loadingOverlay.style.display = 'flex';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('perspective_type', currentPerspective);
    
    console.log(`Sending request with perspective type: ${currentPerspective}`);
    console.log(`File: ${selectedFile.name}, Size: ${(selectedFile.size / 1024).toFixed(2)} KB`);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        console.log(`Response status: ${response.status}`);
        
        // Check if response is OK
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            let errorMessage = `Server error (${response.status})`;
            try {
                const errorJson = JSON.parse(errorText);
                errorMessage = errorJson.error || errorMessage;
            } catch(e) {
                errorMessage = errorText.substring(0, 200) || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            currentResultPath = data.result_image;
            resultImage.src = data.result_image + '?t=' + Date.now();
            
            // Build info panel
            let infoHtml = `
                <div class="info-item">
                    <h4>📐 Perspective Type</h4>
                    <div class="value" style="font-size: 1.1rem;">${data.data.perspective_type}</div>
                </div>
                <div class="info-item">
                    <h4>📈 Lines Detected</h4>
                    <div class="value">${data.data.num_lines || 'N/A'}</div>
                    <div class="small">used for analysis</div>
                </div>
                <div class="info-item">
                    <h4>🎯 Vanishing Points</h4>
                    <div class="value">${data.data.vanishing_points.length}</div>
                    <div class="small">detected</div>
                </div>
            `;
            
            data.data.vanishing_points.forEach((vp, idx) => {
                infoHtml += `
                    <div class="info-item">
                        <h4>📍 ${vp.label || `VP${idx + 1}`}</h4>
                        <div class="value">(${vp.x}, ${vp.y})</div>
                        ${vp.confidence ? `<div class="small">confidence: ${vp.confidence}</div>` : ''}
                    </div>
                `;
            });
            
            if (data.data.horizon_line && data.data.horizon_line.slope !== undefined) {
                infoHtml += `
                    <div class="info-item">
                        <h4>🌊 Horizon Line</h4>
                        <div class="value">Slope: ${data.data.horizon_line.slope}</div>
                        <div class="small">Connects VP1 & VP2</div>
                    </div>
                `;
            }
            
            infoPanel.innerHTML = infoHtml;
            resultsSection.style.display = 'block';
            
            // Remove any existing error
            const oldError = document.querySelector('.error-message');
            if (oldError) oldError.remove();
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Fetch error:', error);
        showError(error.message || 'Network error. Check if server is running at http://127.0.0.1:5000');
    } finally {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
    }
});

// Download button
downloadBtn.addEventListener('click', () => {
    if (currentResultPath) {
        const link = document.createElement('a');
        link.href = currentResultPath;
        link.download = `perspective_analysis_${new Date().getTime()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        console.log('Download started:', currentResultPath);
    } else {
        showError('No result image available to download');
    }
});

// Keyboard shortcut: Enter to analyze (if file selected)
document.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && selectedFile && !analyzeBtn.disabled) {
        analyzeBtn.click();
    }
});

// Reset button functionality (optional - add a reset button if needed)
function resetApp() {
    selectedFile = null;
    currentResultPath = '';
    previewCard.style.display = 'none';
    previewImage.src = '';
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.5';
    fileInput.value = '';
    
    // Remove any errors
    const oldError = document.querySelector('.error-message');
    if (oldError) oldError.remove();
    
    console.log('App reset');
}

// Log when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Script loaded successfully');
    console.log('Upload area:', uploadArea);
    console.log('Analyze button:', analyzeBtn);
    console.log('Allowed file types:', ALLOWED_EXTENSIONS.join(', '));
});
