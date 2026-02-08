// API Base URL
const API_BASE = window.location.origin;

// State
let currentJobId = null;
let currentSourceType = null; // "url" | "file" from upload response
let statusPollInterval = null;

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const uploadSection = document.getElementById('upload-section');
const statusSection = document.getElementById('status-section');
const resultSection = document.getElementById('result-section');
const errorSection = document.getElementById('error-section');
const submitBtn = document.getElementById('submit-btn');
const fileInput = document.getElementById('video-file');
const fileName = document.getElementById('file-name');
const statusText = document.getElementById('status-text');
const statusMessage = document.getElementById('status-message');
const progressFill = document.getElementById('progress-fill');
const resultVideo = document.getElementById('result-video');
const videoPlaceholder = document.getElementById('video-placeholder');
const downloadBtn = document.getElementById('download-btn');
const downloadOriginalBtnStatus = document.getElementById('download-original-btn-status');
const downloadOriginalBtnResult = document.getElementById('download-original-btn-result');
const statusDownloadOriginalWrap = document.getElementById('status-download-original');
const errorMessage = document.getElementById('error-message');
const downloadUrlBtn = document.getElementById('download-url-btn');
const videoUrlInput = document.getElementById('video-url');

// Enable/disable "Download video" button next to URL based on valid URL
function updateDownloadUrlButton() {
    if (!downloadUrlBtn || !videoUrlInput) return;
    const url = (videoUrlInput.value || '').trim();
    let valid = false;
    try {
        if (url) valid = !!new URL(url);
    } catch (_) {}
    downloadUrlBtn.disabled = !valid;
}

// Toggle between file and URL input
function toggleSource() {
    const sourceFile = document.getElementById('source-file').checked;
    const fileGroup = document.getElementById('file-upload-group');
    const urlGroup = document.getElementById('url-input-group');
    const fileInput = document.getElementById('video-file');
    const urlInput = document.getElementById('video-url');
    
    if (sourceFile) {
        fileGroup.style.display = 'block';
        urlGroup.style.display = 'none';
        fileInput.required = true;
        urlInput.required = false;
        urlInput.value = ''; // Clear URL when switching to file
    } else {
        fileGroup.style.display = 'none';
        urlGroup.style.display = 'block';
        fileInput.required = false;
        urlInput.required = true;
        fileInput.value = ''; // Clear file when switching to URL
        fileName.textContent = 'Choose a file...';
        fileName.style.color = 'var(--text-secondary)';
        updateDownloadUrlButton();
    }
}

// Make toggleSource available globally
window.toggleSource = toggleSource;

// URL input: update Download video button state when user types/pastes
if (videoUrlInput) {
    videoUrlInput.addEventListener('input', updateDownloadUrlButton);
    videoUrlInput.addEventListener('change', updateDownloadUrlButton);
}

// Download video from URL (no Create Reel) — button next to Video URL
if (downloadUrlBtn) {
    downloadUrlBtn.addEventListener('click', async () => {
        const url = (videoUrlInput && videoUrlInput.value) ? videoUrlInput.value.trim() : '';
        if (!url) return;
        try {
            downloadUrlBtn.disabled = true;
            downloadUrlBtn.textContent = '⏳ Downloading...';
            const response = await fetch(`${API_BASE}/download-from-url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_url: url })
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `Download failed (${response.status})`);
            }
            const blob = await response.blob();
            const objectUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = objectUrl;
            a.download = 'downloaded_reel.mp4';
            a.click();
            URL.revokeObjectURL(objectUrl);
        } catch (error) {
            console.error('Download from URL error:', error);
            alert(error.message || 'Failed to download video from URL. Check the URL and try again.');
        } finally {
            downloadUrlBtn.disabled = false;
            downloadUrlBtn.textContent = '📥 Download video';
            updateDownloadUrlButton();
        }
    });
}

// File input handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = file.name;
        fileName.style.color = 'var(--text-primary)';
    } else {
        fileName.textContent = 'Choose a file...';
        fileName.style.color = 'var(--text-secondary)';
    }
});

// Product images file handler
const productImagesInput = document.getElementById('product-images');
const productImagesList = document.getElementById('product-images-list');
if (productImagesInput && productImagesList) {
    productImagesInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const fileList = Array.from(files).map((file, idx) => {
                const size = (file.size / 1024 / 1024).toFixed(2);
                return `${idx + 1}. ${file.name} (${size} MB)`;
            }).join('<br>');
            productImagesList.innerHTML = `<strong style="color: var(--primary-color);">${files.length} file(s) selected:</strong><br>${fileList}`;
            productImagesList.style.color = 'var(--text-primary)';
        } else {
            productImagesList.innerHTML = '';
        }
    });
}

// Product logo file handler
const productLogoInput = document.getElementById('product-logo');
const productLogoName = document.getElementById('product-logo-name');
if (productLogoInput && productLogoName) {
    productLogoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const size = (file.size / 1024 / 1024).toFixed(2);
            productLogoName.innerHTML = `<strong style="color: var(--primary-color);">Selected:</strong> ${file.name} (${size} MB)`;
            productLogoName.style.color = 'var(--text-primary)';
        } else {
            productLogoName.innerHTML = '';
        }
    });
}

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const sourceFile = document.getElementById('source-file').checked;
    const productDescription = document.getElementById('product-description').value;
    const brandTone = document.getElementById('brand-tone').value;
    
    // Handle file upload
    if (sourceFile) {
        const videoFile = fileInput.files[0];
        if (!videoFile) {
            showError('Please select a video file');
            return;
        }
        
        // Validate file size (50MB max)
        if (videoFile.size > 50 * 1024 * 1024) {
            showError('File size must be less than 50MB');
            return;
        }
        
        formData.append('video_file', videoFile);
    } else {
        // Handle URL input
        const videoUrl = document.getElementById('video-url').value;
        if (!videoUrl) {
            showError('Please enter a video URL');
            return;
        }
        
        // Basic URL validation
        try {
            new URL(videoUrl);
        } catch {
            showError('Please enter a valid URL');
            return;
        }
        
        formData.append('video_url', videoUrl);
    }
    
    formData.append('product_description', productDescription);
    if (brandTone) {
        formData.append('brand_tone', brandTone);
    }
    
    // Add product images if provided
    const productImages = document.getElementById('product-images');
    if (productImages && productImages.files && productImages.files.length > 0) {
        for (let i = 0; i < productImages.files.length; i++) {
            formData.append('product_images', productImages.files[i]);
        }
    }
    
    // Add product logo if provided
    const productLogo = document.getElementById('product-logo');
    if (productLogo && productLogo.files && productLogo.files.length > 0) {
        formData.append('product_logo', productLogo.files[0]);
    }
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline-block';
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        currentSourceType = data.source_type || null;
        
        // Hide upload form, show status
        uploadSection.style.display = 'none';
        statusSection.style.display = 'block';
        errorSection.style.display = 'none';
        
        // Show "Download original video" in status section when source was URL (available immediately)
        if (currentSourceType === 'url' && currentJobId) {
            statusDownloadOriginalWrap.style.display = 'flex';
            downloadOriginalBtnStatus.href = `${API_BASE}/download/${currentJobId}/original`;
            downloadOriginalBtnStatus.download = `${currentJobId}_original.mp4`;
        } else {
            statusDownloadOriginalWrap.style.display = 'none';
        }
        
        // Start polling for status
        startStatusPolling(currentJobId);
        
    } catch (error) {
        console.error('Upload error:', error);
        showError(error.message || 'Failed to upload video. Please try again.');
        resetSubmitButton();
    }
});

// Status polling
function startStatusPolling(jobId) {
    statusText.textContent = 'Processing...';
    statusText.className = 'status-value status-processing';
    statusMessage.textContent = 'Extracting style, generating blueprint, and rendering video...';
    progressFill.style.width = '30%';
    
    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/status/${jobId}`);
            const data = await response.json();
            
            updateStatus(data);
            
            if (data.status === 'completed') {
                clearInterval(statusPollInterval);
                showResult(jobId, data);
            } else if (data.status === 'failed') {
                clearInterval(statusPollInterval);
                showError(data.error || 'Processing failed. Please try again.');
            }
        } catch (error) {
            console.error('Status check error:', error);
            // Continue polling on error
        }
    }, 2000); // Poll every 2 seconds
}

// Update status display
function updateStatus(data) {
    const status = data.status;
    
    switch (status) {
        case 'processing':
            statusText.textContent = 'Processing...';
            statusMessage.textContent = 'Extracting style, generating blueprint, and rendering video...';
            progressFill.style.width = '50%';
            break;
        case 'completed':
            statusText.textContent = 'Completed!';
            statusText.className = 'status-value';
            statusMessage.textContent = 'Your reel is ready!';
            progressFill.style.width = '100%';
            break;
        case 'failed':
            statusText.textContent = 'Failed';
            statusText.className = 'status-value';
            statusMessage.textContent = data.error || 'Processing failed';
            progressFill.style.width = '0%';
            break;
        default:
            statusText.textContent = status;
            progressFill.style.width = '30%';
    }
}

// Show result
async function showResult(jobId, jobData) {
    statusSection.style.display = 'none';
    resultSection.style.display = 'block';
    errorSection.style.display = 'none';
    
    const sourceType = (jobData && jobData.source_type) || currentSourceType;
    
    // Set download link for rendered reel
    downloadBtn.href = `${API_BASE}/download/${jobId}`;
    downloadBtn.download = `${jobId}_reel.mp4`;
    
    // Show "Download original video" on result when source was URL
    if (sourceType === 'url') {
        downloadOriginalBtnResult.style.display = 'inline-flex';
        downloadOriginalBtnResult.href = `${API_BASE}/download/${jobId}/original`;
        downloadOriginalBtnResult.download = `${jobId}_original.mp4`;
    } else {
        downloadOriginalBtnResult.style.display = 'none';
    }
    
    // Try to load video preview
    try {
        const videoUrl = `${API_BASE}/download/${jobId}`;
        resultVideo.src = videoUrl;
        resultVideo.style.display = 'block';
        videoPlaceholder.style.display = 'none';
    } catch (error) {
        console.error('Video preview error:', error);
        videoPlaceholder.textContent = 'Video ready for download';
    }
}

// Show error
function showError(message) {
    uploadSection.style.display = 'none';
    statusSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
    
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
    
    resetSubmitButton();
}

// Reset form
function resetForm() {
    uploadForm.reset();
    fileName.textContent = 'Choose a file...';
    fileName.style.color = 'var(--text-secondary)';
    
    // Clear product images and logo displays
    if (productImagesList) productImagesList.innerHTML = '';
    if (productLogoName) productLogoName.innerHTML = '';
    if (productImagesInput) productImagesInput.value = '';
    if (productLogoInput) productLogoInput.value = '';
    
    // Reset to file upload mode
    document.getElementById('source-file').checked = true;
    toggleSource();
    
    uploadSection.style.display = 'block';
    statusSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    resultVideo.src = '';
    resultVideo.style.display = 'none';
    videoPlaceholder.style.display = 'block';
    
    currentJobId = null;
    currentSourceType = null;
    if (downloadOriginalBtnResult) {
        downloadOriginalBtnResult.style.display = 'none';
        downloadOriginalBtnResult.removeAttribute('href');
    }
    if (statusDownloadOriginalWrap) statusDownloadOriginalWrap.style.display = 'none';
    progressFill.style.width = '0%';
    
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
    
    resetSubmitButton();
}

// Reset submit button
function resetSubmitButton() {
    submitBtn.disabled = false;
    submitBtn.querySelector('.btn-text').style.display = 'inline';
    submitBtn.querySelector('.btn-loader').style.display = 'none';
}

// Make resetForm available globally
window.resetForm = resetForm;


