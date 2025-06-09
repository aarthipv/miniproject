// Historical Text Reconstruction - JavaScript Application

class TextReconstructionApp {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.loadExamples();
        this.initializeTheme();
    }

    initializeElements() {
        // Form elements
        this.inputText = document.getElementById('inputText');
        this.reconstructBtn = document.getElementById('reconstructBtn');
        this.clearBtn = document.getElementById('clearBtn');
        
        // Result elements
        this.resultsSection = document.getElementById('resultsSection');
        this.originalText = document.getElementById('originalText');
        this.reconstructedText = document.getElementById('reconstructedText');
        this.highlightedText = document.getElementById('highlightedText');
        
        // UI elements
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.errorAlert = document.getElementById('errorAlert');
        this.errorMessage = document.getElementById('errorMessage');
        this.examplesMenu = document.getElementById('examplesMenu');
        
        // Theme elements
        this.themeToggle = document.getElementById('themeToggle');
        this.themeIcon = document.getElementById('themeIcon');
    }

    bindEvents() {
        // Main functionality
        this.reconstructBtn.addEventListener('click', () => this.reconstructText());
        this.clearBtn.addEventListener('click', () => this.clearForm());
        this.inputText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.reconstructText();
            }
        });

        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // Auto-resize textarea
        this.inputText.addEventListener('input', () => this.autoResizeTextarea());
    }

    async loadExamples() {
        try {
            const response = await fetch('/examples');
            if (!response.ok) {
                throw new Error('Failed to load examples');
            }
            
            const examples = await response.json();
            this.populateExamplesMenu(examples);
        } catch (error) {
            console.error('Error loading examples:', error);
            this.examplesMenu.innerHTML = '<li><a class="dropdown-item" href="#">Examples unavailable</a></li>';
        }
    }

    populateExamplesMenu(examples) {
        this.examplesMenu.innerHTML = '';
        
        examples.forEach((example, index) => {
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.className = 'dropdown-item';
            link.href = '#';
            link.innerHTML = `
                <strong>${example.title}</strong><br>
                <small class="text-muted">${example.description}</small>
            `;
            
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.loadExample(example);
            });
            
            listItem.appendChild(link);
            this.examplesMenu.appendChild(listItem);
            
            // Add divider between examples (except after last one)
            if (index < examples.length - 1) {
                const divider = document.createElement('li');
                divider.innerHTML = '<hr class="dropdown-divider">';
                this.examplesMenu.appendChild(divider);
            }
        });
    }

    loadExample(example) {
        this.inputText.value = example.damaged;
        this.autoResizeTextarea();
        this.hideError();
        this.hideResults();
        
        // Add visual feedback
        this.inputText.classList.add('pulse');
        setTimeout(() => {
            this.inputText.classList.remove('pulse');
        }, 1000);
    }

    async reconstructText() {
        const text = this.inputText.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to reconstruct.');
            return;
        }

        this.showLoading();
        this.hideError();
        this.hideResults();

        try {
            const response = await fetch('/reconstruct', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Reconstruction failed');
            }

            const result = await response.json();
            this.displayResults(result);
            
        } catch (error) {
            console.error('Reconstruction error:', error);
            this.showError(`Reconstruction failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Populate result areas
        this.originalText.textContent = result.original;
        this.reconstructedText.textContent = result.reconstructed;
        this.highlightedText.innerHTML = result.highlighted;

        // Show results with animation
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('fade-in-up');
        
        // Scroll to results
        setTimeout(() => {
            this.resultsSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
    }

    clearForm() {
        this.inputText.value = '';
        this.hideResults();
        this.hideError();
        this.autoResizeTextarea();
        this.inputText.focus();
    }

    showLoading() {
        this.loadingIndicator.style.display = 'block';
        this.reconstructBtn.disabled = true;
        this.reconstructBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Reconstructing...';
    }

    hideLoading() {
        this.loadingIndicator.style.display = 'none';
        this.reconstructBtn.disabled = false;
        this.reconstructBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Reconstruct Text';
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorAlert.style.display = 'block';
        this.errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    hideError() {
        this.errorAlert.style.display = 'none';
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
        this.resultsSection.classList.remove('fade-in-up');
    }

    autoResizeTextarea() {
        this.inputText.style.height = 'auto';
        this.inputText.style.height = Math.max(this.inputText.scrollHeight, 100) + 'px';
    }

    // Theme management
    initializeTheme() {
        const savedTheme = localStorage.getItem('textReconstructionTheme') || 'light';
        this.setTheme(savedTheme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('textReconstructionTheme', theme);
        
        // Update theme icon
        if (theme === 'dark') {
            this.themeIcon.className = 'fas fa-moon';
            this.themeToggle.title = 'Switch to light mode';
        } else {
            this.themeIcon.className = 'fas fa-sun';
            this.themeToggle.title = 'Switch to dark mode';
        }
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new TextReconstructionApp();
    
    // Add some visual polish
    setTimeout(() => {
        document.body.classList.add('fade-in-up');
    }, 100);
    
    // Add keyboard shortcuts info to console
    console.log('Historical Text Reconstruction App loaded successfully!');
    console.log('Keyboard shortcuts:');
    console.log('- Ctrl+Enter: Reconstruct text');
    console.log('- Theme toggle: Click the sun/moon icon');
});

// Service Worker registration (optional, for offline functionality)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Service worker implementation can be added here if needed
    });
}
