// Configuration - get from config.js
const API_URL = window.APP_CONFIG ? window.APP_CONFIG.API_URL : 'http://localhost:8000';

// DOM Elements
const elements = {
    userStory: document.getElementById('userStory'),
    testType: document.getElementById('testType'),
    maxLength: document.getElementById('maxLength'),
    sliderValue: document.getElementById('sliderValue'),
    useTemplate: document.getElementById('useTemplate'),
    charCount: document.getElementById('charCount'),
    generateBtn: document.getElementById('generateBtn'),
    clearBtn: document.getElementById('clearBtn'),
    copyBtn: document.getElementById('copyBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    resultArea: document.getElementById('resultArea'),
    metadata: document.getElementById('metadata'),
    metaTestType: document.getElementById('metaTestType'),
    metaConfidence: document.getElementById('metaConfidence'),
    metaTime: document.getElementById('metaTime'),
    metaModel: document.getElementById('metaModel'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    toastContainer: document.getElementById('toastContainer')
};

// State
let currentTestCase = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
});

/**
 * Khởi tạo các event listeners
 */
function initializeEventListeners() {
    // User story character count
    elements.userStory.addEventListener('input', (e) => {
        const length = e.target.value.length;
        elements.charCount.textContent = `${length} / 500`;
        
        // Limit to 500 characters
        if (length > 500) {
            e.target.value = e.target.value.substring(0, 500);
            elements.charCount.textContent = '500 / 500';
        }
    });

    // Slider value update
    elements.maxLength.addEventListener('input', (e) => {
        elements.sliderValue.textContent = e.target.value;
    });

    // Generate button
    elements.generateBtn.addEventListener('click', generateTestCase);

    // Clear button
    elements.clearBtn.addEventListener('click', clearForm);

    // Copy button
    elements.copyBtn.addEventListener('click', copyTestCase);

    // Download button
    elements.downloadBtn.addEventListener('click', downloadTestCase);

    // Sample chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', (e) => {
            const story = e.currentTarget.getAttribute('data-story');
            elements.userStory.value = story;
            elements.charCount.textContent = `${story.length} / 500`;
        });
    });

    // Enter key to generate
    elements.userStory.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            generateTestCase();
        }
    });
}

/**
 * Kiểm tra API health
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('API Status:', data);
            
            // Load model info
            loadModelInfo();
        } else {
            showToast('API is not responding. Please start the server.', 'error');
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showToast('Cannot connect to API server. Please ensure server.py is running.', 'error');
    }
}

/**
 * Load thông tin model
 */
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/api/model_info`);
        if (response.ok) {
            const data = await response.json();
            console.log('Model Info:', data);
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

/**
 * Generate test case từ user story
 */
async function generateTestCase() {
    // Validate input
    const userStory = elements.userStory.value.trim();
    if (!userStory) {
        showToast('Please enter a user story', 'error');
        elements.userStory.focus();
        return;
    }

    if (userStory.length < 10) {
        showToast('User story is too short (minimum 10 characters)', 'error');
        return;
    }

    // Show loading
    showLoading(true);
    elements.generateBtn.disabled = true;

    // Prepare request data
    const requestData = {
        user_story: userStory,
        test_type: elements.testType.value,
        max_length: parseInt(elements.maxLength.value),
        use_template: elements.useTemplate.checked
    };

    try {
        // Call API
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // Display result
            displayTestCase(data.test_case);
            
            // Display metadata
            displayMetadata(data.metadata);
            
            // Enable action buttons
            elements.copyBtn.disabled = false;
            elements.downloadBtn.disabled = false;
            
            // Store current test case
            currentTestCase = stripHtml(data.test_case);
            
            showToast('Test case generated successfully!', 'success');
        } else {
            throw new Error(data.error || 'Failed to generate test case');
        }
    } catch (error) {
        console.error('Error generating test case:', error);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
        elements.generateBtn.disabled = false;
    }
}

/**
 * Hiển thị test case
 */
function displayTestCase(htmlContent) {
    // Parse và highlight keywords
    const highlighted = highlightGherkinKeywords(htmlContent);
    
    // Display trong result area
    elements.resultArea.innerHTML = `
        <div class="test-output">
            ${highlighted}
        </div>
    `;
    
    // Scroll to result
    elements.resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Highlight Gherkin keywords
 */
function highlightGherkinKeywords(html) {
    // Keywords to highlight
    const keywords = {
        'Given': '#53e3a6',
        'When': '#53e3a6',
        'Then': '#53e3a6',
        'And': '#53e3a6',
        'But': '#53e3a6',
        'Feature:': '#ffd700',
        'Scenario:': '#ffd700',
        'Test ID:': '#4299e1',
        'Test Case:': '#4299e1'
    };
    
    let result = html;
    
    // Highlight each keyword
    Object.entries(keywords).forEach(([keyword, color]) => {
        const regex = new RegExp(`\\b(${keyword})\\b`, 'g');
        result = result.replace(regex, `<span style="color: ${color}; font-weight: bold;">$1</span>`);
    });
    
    return result;
}

/**
 * Hiển thị metadata
 */
function displayMetadata(metadata) {
    if (!metadata) return;
    
    elements.metadata.classList.remove('hidden');
    
    // Test type
    elements.metaTestType.textContent = formatTestType(metadata.test_type || 'N/A');
    
    // Confidence
    const confidence = metadata.confidence || 0;
    elements.metaConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;
    
    // Generation time
    const time = metadata.generation_time || 0;
    elements.metaTime.textContent = `${time.toFixed(2)}s`;
    
    // Model info
    elements.metaModel.textContent = metadata.model || 'Fine-tuned Model';
}

/**
 * Format test type cho display
 */
function formatTestType(type) {
    const types = {
        'functional': 'Functional',
        'security': 'Security',
        'performance': 'Performance',
        'compliance': 'Compliance'
    };
    return types[type] || type;
}

/**
 * Clear form
 */
function clearForm() {
    elements.userStory.value = '';
    elements.charCount.textContent = '0 / 500';
    elements.testType.value = 'functional';
    elements.maxLength.value = 150;
    elements.sliderValue.textContent = '150';
    elements.useTemplate.checked = true;
    
    // Clear result
    elements.resultArea.innerHTML = `
        <div class="empty-state">
            <svg width="60" height="60" viewBox="0 0 60 60" fill="none" opacity="0.5">
                <rect x="10" y="15" width="40" height="30" rx="3" stroke="currentColor" stroke-width="2"/>
                <path d="M20 25H40M20 30H35M20 35H38" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <p>Generated test cases will appear here</p>
            <p class="text-muted">Enter a user story and click Generate to begin</p>
        </div>
    `;
    
    // Hide metadata
    elements.metadata.classList.add('hidden');
    
    // Disable action buttons
    elements.copyBtn.disabled = true;
    elements.downloadBtn.disabled = true;
    
    currentTestCase = '';
    
    showToast('Form cleared', 'info');
}

/**
 * Copy test case vào clipboard
 */
async function copyTestCase() {
    if (!currentTestCase) return;
    
    try {
        await navigator.clipboard.writeText(currentTestCase);
        showToast('Test case copied to clipboard!', 'success');
        
        // Visual feedback
        elements.copyBtn.style.color = '#48bb78';
        setTimeout(() => {
            elements.copyBtn.style.color = '';
        }, 2000);
    } catch (error) {
        console.error('Failed to copy:', error);
        showToast('Failed to copy to clipboard', 'error');
    }
}

/**
 * Download test case dạng file
 */
function downloadTestCase() {
    if (!currentTestCase) return;
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `test_case_${timestamp}.txt`;
    
    const blob = new Blob([currentTestCase], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showToast(`Downloaded as ${filename}`, 'success');
}

/**
 * Strip HTML tags từ content
 */
function stripHtml(html) {
    const temp = document.createElement('div');
    temp.innerHTML = html;
    return temp.textContent || temp.innerText || '';
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    if (show) {
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => {
            elements.toastContainer.removeChild(toast);
        }, 300);
    }, 3000);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter to generate
    if (e.ctrlKey && e.key === 'Enter' && document.activeElement === elements.userStory) {
        generateTestCase();
    }
    
    // Escape to clear
    if (e.key === 'Escape') {
        clearForm();
    }
    
    // Ctrl+C to copy (when result is available)
    if (e.ctrlKey && e.key === 'c' && currentTestCase && !window.getSelection().toString()) {
        copyTestCase();
    }
});

// Console branding
console.log('%c🏦 Banking Test Case Generator', 'color: #667eea; font-size: 20px; font-weight: bold');
console.log('%cPowered by Your Fine-tuned Model (8 epochs, 8000+ samples)', 'color: #a0aec0');
console.log('%cMaster\'s Thesis - Vu Tuan Chien', 'color: #718096');
