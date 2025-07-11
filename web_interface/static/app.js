/**
 * Wildfire Watch Status Panel - Minimal JavaScript
 * 
 * This file contains only essential functionality for the web interface.
 * We use HTMX for most interactivity to keep the JavaScript footprint small.
 */

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Update timestamp on page load
    updateLastUpdateTime();
    
    // Set up auto-refresh if configured
    setupAutoRefresh();
    
    // Initialize event filtering if on dashboard
    initializeEventFiltering();
    
    // Add click handlers for better UX
    addClickHandlers();
});

/**
 * Update the last update timestamp in the header
 */
function updateLastUpdateTime() {
    const element = document.querySelector('#last-update span');
    if (element) {
        const now = new Date();
        element.textContent = now.toTimeString().split(' ')[0];
    }
}

/**
 * Set up auto-refresh based on configured interval
 */
function setupAutoRefresh() {
    // HTMX will handle most updates, this is for full page refresh
    const refreshInterval = parseInt(document.body.dataset.refreshInterval || '0');
    if (refreshInterval > 0) {
        setTimeout(() => {
            // Only refresh if not in debug mode
            if (!window.location.pathname.includes('/debug')) {
                window.location.reload();
            }
        }, refreshInterval * 1000);
    }
}

/**
 * Initialize event filtering functionality
 */
function initializeEventFiltering() {
    const filterSelect = document.querySelector('select[x-model="filter"]');
    if (filterSelect) {
        filterSelect.addEventListener('change', function() {
            const filter = this.value;
            const items = document.querySelectorAll('.event-item');
            
            items.forEach(item => {
                if (filter === 'all' || item.dataset.type === filter) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
}

/**
 * Add click handlers for better UX
 */
function addClickHandlers() {
    // Copy to clipboard for event details
    document.querySelectorAll('pre').forEach(pre => {
        pre.style.cursor = 'pointer';
        pre.title = 'Click to copy';
        pre.addEventListener('click', function() {
            copyToClipboard(this.textContent);
        });
    });
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy:', err);
        });
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy:', err);
        }
        document.body.removeChild(textarea);
    }
}

/**
 * Show a temporary notification
 */
function showNotification(message) {
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 2000);
}

/**
 * Format timestamps for display
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Sanitize HTML to prevent XSS
 */
function sanitizeHTML(html) {
    const temp = document.createElement('div');
    temp.textContent = html;
    return temp.innerHTML;
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Export functions for use in inline scripts
window.updateLastUpdateTime = updateLastUpdateTime;
window.showNotification = showNotification;
window.sanitizeHTML = sanitizeHTML;