/**
 * COSMIC FINANCE - Neural Price Predictor
 * Advanced price prediction interface with multiple neural network models
 * Version: 2.0.1
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const predictorForm = document.getElementById('predictor-form');
    const tickerInput = document.getElementById('ticker');
    const tickerSuggestions = document.getElementById('ticker-suggestions');
    const futureDateInput = document.getElementById('future_date');
    const searchTickerButton = document.getElementById('search-ticker');
    const generateButton = document.getElementById('generate-prediction');
    const modelToggles = document.querySelectorAll('input[name="models"]');
    const chartTabs = document.querySelectorAll('.chart-tab');
    const charts = document.querySelectorAll('.chart');
    
    // Initialize cosmic interface elements
    initializeCosmicInterface();
    
    // Set default date (30 days from today) if not already set
    if (futureDateInput && !futureDateInput.value) {
        const today = new Date();
        const futureDate = new Date(today);
        futureDate.setDate(today.getDate() + 30);
        
        const formattedDate = futureDate.toISOString().split('T')[0];
        futureDateInput.value = formattedDate;
        futureDateInput.min = new Date().toISOString().split('T')[0];
    }
    
    // Event Listeners
    if (searchTickerButton) {
        searchTickerButton.addEventListener('click', searchTicker);
    }
    
    if (tickerInput) {
        // Ticker input events
        tickerInput.addEventListener('input', debounce(handleTickerInput, 300));
        tickerInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                validateTicker();
            }
        });
        
        // Blur event to hide suggestions when clicking outside
        document.addEventListener('click', function(e) {
            if (e.target !== tickerInput && e.target !== tickerSuggestions) {
                if (tickerSuggestions) tickerSuggestions.style.display = 'none';
            }
        });
    }
    
    if (predictorForm) {
        predictorForm.addEventListener('submit', handleFormSubmit);
    }
    
    // Model toggle validation
    modelToggles.forEach(toggle => {
        toggle.addEventListener('change', validateModelSelection);
    });
    
    // Chart tab functionality
    if (chartTabs.length > 0) {
        chartTabs.forEach(tab => {
            tab.addEventListener('click', function() {
                const chartId = this.getAttribute('data-chart');
                
                // Update active tab
                chartTabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                // Update active chart
                charts.forEach(chart => {
                    chart.classList.remove('active');
                    if (chart.id === chartId + '-chart') {
                        chart.classList.add('active');
                        
                        // Trigger Plotly resize for the chart that's now visible
                        if (window.Plotly && chart.querySelector('.js-plotly-plot')) {
                            window.Plotly.Plots.resize(chart);
                        }
                    }
                });
            });
        });
    }
    
    // Window resize event for responsive charts
    window.addEventListener('resize', function() {
        if (window.Plotly) {
            const activeChart = document.querySelector('.chart.active');
            if (activeChart && activeChart.querySelector('.js-plotly-plot')) {
                window.Plotly.Plots.resize(activeChart);
            }
        }
    });
    
    // Initialize Plotly charts if data exists
    initializeCharts();
    
    /**
     * Initialize the cosmic interface elements with animations and effects
     */
    function initializeCosmicInterface() {
        // Add glitch text effect to headings
        const glitchTexts = document.querySelectorAll('.glitch-text');
        glitchTexts.forEach(el => {
            if (!el.getAttribute('data-text')) {
                el.setAttribute('data-text', el.textContent);
            }
        });
        
        // Add neural network background animation if canvas exists
        const galaxyCanvas = document.getElementById('galaxyCanvas');
        if (galaxyCanvas) {
            initNeuralBackground(galaxyCanvas);
        }
        
        // Add pulse effect to prediction values
        const predictionValues = document.querySelectorAll('.price-prediction');
        predictionValues.forEach(el => {
            el.classList.add('pulse-glow');
        });
        
        // Initialize tooltips
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(el => {
            el.addEventListener('mouseenter', showTooltip);
            el.addEventListener('mouseleave', hideTooltip);
        });
    }
    
    /**
     * Initialize neural network background animation
     * @param {HTMLElement} canvas - The canvas element
     */
    function initNeuralBackground(canvas) {
        const ctx = canvas.getContext('2d');
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Set canvas dimensions
        canvas.width = width;
        canvas.height = height;
        
        // Neural node properties
        const nodes = [];
        const nodeCount = Math.min(100, Math.floor(width * height / 10000));
        const connections = [];
        
        // Initialize nodes
        for (let i = 0; i < nodeCount; i++) {
            nodes.push({
                x: Math.random() * width,
                y: Math.random() * height,
                radius: Math.random() * 2 + 1,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                color: `rgba(15, 248, 231, ${Math.random() * 0.5 + 0.2})`
            });
        }
        
        // Create connections between nodes
        for (let i = 0; i < nodeCount; i++) {
            for (let j = i + 1; j < nodeCount; j++) {
                if (Math.random() > 0.97) {
                    connections.push({
                        from: i,
                        to: j,
                        width: Math.random() * 0.3 + 0.1,
                        enabled: Math.random() > 0.5,
                        pulseTime: 0,
                        pulseSpeed: Math.random() * 0.01 + 0.005
                    });
                }
            }
        }
        
        // Animation function
        function animate() {
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = 'rgba(10, 10, 21, 0.1)';
            ctx.fillRect(0, 0, width, height);
            
            // Update and draw connections
            for (let i = 0; i < connections.length; i++) {
                const connection = connections[i];
                const fromNode = nodes[connection.from];
                const toNode = nodes[connection.to];
                
                // Calculate distance between nodes
                const dx = toNode.x - fromNode.x;
                const dy = toNode.y - fromNode.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Only draw connections within a certain distance
                if (distance < 150) {
                    // Update pulse for data flow visualization
                    connection.pulseTime += connection.pulseSpeed;
                    if (connection.pulseTime > 1) connection.pulseTime = 0;
                    
                    // Draw connection line
                    ctx.beginPath();
                    ctx.moveTo(fromNode.x, fromNode.y);
                    ctx.lineTo(toNode.x, toNode.y);
                    
                    // Draw with opacity based on distance
                    const opacity = Math.max(0, (150 - distance) / 150 * 0.5);
                    ctx.strokeStyle = `rgba(15, 248, 231, ${opacity})`;
                    ctx.lineWidth = connection.width;
                    ctx.stroke();
                    
                    // Draw data pulse if connection is enabled
                    if (connection.enabled) {
                        const pulsePosition = {
                            x: fromNode.x + dx * connection.pulseTime,
                            y: fromNode.y + dy * connection.pulseTime
                        };
                        
                        ctx.beginPath();
                        ctx.arc(pulsePosition.x, pulsePosition.y, 1.5, 0, Math.PI * 2);
                        ctx.fillStyle = 'rgba(15, 248, 231, 0.8)';
                        ctx.fill();
                    }
                }
            }
            
            // Update and draw nodes
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                
                // Update position
                node.x += node.vx;
                node.y += node.vy;
                
                // Bounce off edges
                if (node.x < 0 || node.x > width) node.vx *= -1;
                if (node.y < 0 || node.y > height) node.vy *= -1;
                
                // Draw node
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                ctx.fillStyle = node.color;
                ctx.fill();
            }
            
            requestAnimationFrame(animate);
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
        });
        
        // Start animation
        animate();
    }
    
    /**
     * Initialize Plotly charts if data exists
     */
    function initializeCharts() {
        // Check if Plotly is available
        if (!window.Plotly) {
            console.error('Plotly.js not loaded. Please include the Plotly.js library.');
            return;
        }
        
        // Check if chart containers exist
        const predictionChart = document.getElementById('prediction-chart');
        const historicalChart = document.getElementById('historical-chart');
        const comparisonChart = document.getElementById('comparison-chart');
        
        // If no chart containers, exit
        if (!predictionChart && !historicalChart && !comparisonChart) {
            return;
        }
        
        // Function to create a placeholder chart if needed
        function createPlaceholderChart(container, title) {
            if (!container.querySelector('.js-plotly-plot') && !container.hasAttribute('data-chart-loaded')) {
                const layout = {
                    title: {
                        text: title,
                        font: { family: 'Rajdhani, sans-serif', size: 20, color: '#0ff8e7' }
                    },
                    paper_bgcolor: 'rgba(10, 15, 20, 0.6)',
                    plot_bgcolor: 'rgba(10, 15, 20, 0.6)',
                    font: { family: 'Rajdhani, sans-serif', color: '#a4b8c4' },
                    xaxis: {
                        title: 'Date',
                        gridcolor: 'rgba(40, 40, 40, 0.8)',
                        zerolinecolor: 'rgba(40, 40, 40, 0.8)',
                        tickfont: { color: '#8194a9' }
                    },
                    yaxis: {
                        title: 'Price ($)',
                        gridcolor: 'rgba(40, 40, 40, 0.8)',
                        zerolinecolor: 'rgba(40, 40, 40, 0.8)',
                        tickfont: { color: '#8194a9' }
                    },
                    showlegend: true,
                    legend: {
                        font: { family: 'Rajdhani, sans-serif', size: 12, color: '#a4b8c4' }
                    },
                    margin: { l: 50, r: 20, t: 50, b: 50 }
                };
                
                // Create empty plot to be filled with data
                Plotly.newPlot(container, [], layout, {
                    responsive: true,
                    displayModeBar: false
                });
                
                // Mark as placeholder
                container.setAttribute('data-is-placeholder', 'true');
            }
        }
        
        // Create placeholders if needed
        if (predictionChart && !predictionChart.querySelector('.js-plotly-plot')) {
            createPlaceholderChart(predictionChart, 'Price Prediction Forecast');
        }
        
        if (historicalChart && !historicalChart.querySelector('.js-plotly-plot')) {
            createPlaceholderChart(historicalChart, 'Historical Price Data');
        }
        
        if (comparisonChart && !comparisonChart.querySelector('.js-plotly-plot')) {
            createPlaceholderChart(comparisonChart, 'Model Performance Comparison');
        }
        
        // If real chart data is available in the page, it will be loaded via the inline scripts
        // that were added in the template when the data was passed
    }
    
    /**
     * Handle ticker input for auto-suggestions
     */
    function handleTickerInput() {
        const query = tickerInput.value.trim();
        
        if (query.length < 2) {
            if (tickerSuggestions) tickerSuggestions.style.display = 'none';
            return;
        }
        
        // Common stocks for quick suggestions (fallback if API fails)
        const commonStocks = [
            { ticker: 'AAPL', name: 'Apple Inc.' },
            { ticker: 'MSFT', name: 'Microsoft Corporation' },
            { ticker: 'GOOGL', name: 'Alphabet Inc.' },
            { ticker: 'AMZN', name: 'Amazon.com Inc.' },
            { ticker: 'META', name: 'Meta Platforms Inc.' },
            { ticker: 'TSLA', name: 'Tesla Inc.' },
            { ticker: 'NVDA', name: 'NVIDIA Corporation' },
            { ticker: 'JPM', name: 'JPMorgan Chase & Co.' },
            { ticker: 'V', name: 'Visa Inc.' },
            { ticker: 'JNJ', name: 'Johnson & Johnson' },
            { ticker: 'WMT', name: 'Walmart Inc.' },
            { ticker: 'PG', name: 'Procter & Gamble Co.' },
            { ticker: 'MA', name: 'Mastercard Inc.' },
            { ticker: 'HD', name: 'Home Depot Inc.' },
            { ticker: 'BAC', name: 'Bank of America Corp.' },
            { ticker: 'RELIANCE.NS', name: 'Reliance Industries Ltd.' },
            { ticker: 'TCS.NS', name: 'Tata Consultancy Services Ltd.' },
            { ticker: 'INFY', name: 'Infosys Ltd.' }
        ];
        
        // Filter common stocks based on input
        const filteredStocks = commonStocks.filter(stock => {
            return stock.ticker.toLowerCase().includes(query.toLowerCase()) || 
                   stock.name.toLowerCase().includes(query.toLowerCase());
        }).slice(0, 6);  // Limit to 6 suggestions
        
        // Display suggestions
        if (filteredStocks.length > 0 && tickerSuggestions) {
            tickerSuggestions.innerHTML = '';
            filteredStocks.forEach(stock => {
                const suggestion = document.createElement('div');
                suggestion.classList.add('ticker-suggestion');
                suggestion.innerHTML = `<span class="suggestion-ticker">${stock.ticker}</span> <span class="suggestion-name">${stock.name}</span>`;
                suggestion.addEventListener('click', () => {
                    tickerInput.value = stock.ticker;
                    tickerSuggestions.style.display = 'none';
                    validateTicker();
                });
                tickerSuggestions.appendChild(suggestion);
            });
            tickerSuggestions.style.display = 'block';
        } else if (tickerSuggestions) {
            tickerSuggestions.style.display = 'none';
        }
        
        // In a real implementation, you'd call an API here to get stock suggestions
        // Example:
        // fetch(`/api/stock_search?q=${encodeURIComponent(query)}`)
        //     .then(response => response.json())
        //     .then(data => { 
        //         // Display API results instead of the static list
        //     })
        //     .catch(error => {
        //         console.error('Error fetching stock suggestions:', error);
        //     });
    }
    
    /**
     * Search and validate ticker symbol
     */
    function searchTicker() {
        const tickerValue = tickerInput.value.trim().toUpperCase();
        
        if (!tickerValue) {
            showNotification('Please enter a ticker symbol', 'error');
            return;
        }
        
        // Add loading indicator
        if (searchTickerButton) {
            searchTickerButton.classList.add('searching');
        }
        
        // Validate ticker format (allow normal tickers and tickers with extensions like .NS)
        const validTickerFormat = /^[A-Z]{1,5}(\.[A-Z]{1,4})?$/;
        
        if (!validTickerFormat.test(tickerValue)) {
            setTimeout(() => {
                showNotification('Invalid ticker format. Use format like AAPL or RELIANCE.NS', 'error');
                if (searchTickerButton) {
                    searchTickerButton.classList.remove('searching');
                }
            }, 800);
            return;
        }
        
        // For demonstration, we're simulating an API call with a timeout
        setTimeout(() => {
            if (searchTickerButton) {
                searchTickerButton.classList.remove('searching');
            }
            
            // In real implementation, we would verify if the ticker exists
            // For now, assume all tickers are valid
            showNotification(`Ticker ${tickerValue} confirmed`, 'success');
            tickerInput.value = tickerValue;
            
            // Add a validation class for visual feedback
            tickerInput.classList.add('validated');
            setTimeout(() => {
                tickerInput.classList.remove('validated');
            }, 1000);
            
        }, 800);
    }
    
    /**
     * Validate ticker
     */
    function validateTicker() {
        const tickerValue = tickerInput.value.trim();
        
        if (!tickerValue) {
            showNotification('Please enter a ticker symbol', 'error');
            return false;
        }
        
        return true;
    }
    
    /**
     * Handle form submission
     * @param {Event} e - Form submit event
     */
    function handleFormSubmit(e) {
        if (!validateFormInputs()) {
            e.preventDefault();
            return;
        }
        
        // Show loading state on button
        if (generateButton) {
            generateButton.classList.add('loading');
            generateButton.innerHTML = '<span>PROCESSING</span><div class="spinner"></div>';
            generateButton.disabled = true;
        }
        
        // Form will submit naturally
    }
    
    /**
     * Validate all form inputs before submission
     * @returns {boolean} - Whether all inputs are valid
     */
    function validateFormInputs() {
        // 1. Validate ticker
        if (!validateTicker()) {
            return false;
        }
        
        // 2. Validate future date
        const selectedDate = new Date(futureDateInput.value);
        const today = new Date();
        today.setHours(0, 0, 0, 0); // Reset time component for comparison
        
        if (selectedDate <= today) {
            showNotification('Prediction date must be in the future', 'error');
            return false;
        }
        
        // Check if prediction date is too far in the future (reduce accuracy warning)
        const maxDate = new Date(today);
        maxDate.setFullYear(maxDate.getFullYear() + 2); // Max 2 years
        
        if (selectedDate > maxDate) {
            showNotification('Predictions beyond 2 years have reduced accuracy', 'warning');
            // Continue with submission despite warning
        }
        
        // 3. Validate model selection (at least one model must be selected)
        const atLeastOneModelSelected = Array.from(modelToggles).some(toggle => toggle.checked);
        
        if (!atLeastOneModelSelected) {
            showNotification('At least one neural model must be selected', 'error');
            return false;
        }
        
        return true;
    }
    
    /**
     * Validate model selection to ensure at least one is checked
     */
    function validateModelSelection() {
        const atLeastOneChecked = Array.from(modelToggles).some(toggle => toggle.checked);
        
        if (!atLeastOneChecked) {
            // If user tries to uncheck all models, force at least one to stay checked
            const lstmToggle = document.getElementById('lstm');
            if (lstmToggle) {
                lstmToggle.checked = true;
                showNotification('At least one neural model must be selected', 'warning');
            }
        }
    }
    
    /**
     * Show notification message
     * @param {string} message - The notification message
     * @param {string} type - Notification type (success, error, warning, info)
     */
    function showNotification(message, type = 'info') {
        // Remove any existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(n => n.remove());
        
        // Create notification element
        const notification = document.createElement('div');
        notification.classList.add('notification', type);
        
        // Create icon based on type
        const icons = {
            'success': '✓',
            'error': '✗',
            'warning': '⚠',
            'info': 'ℹ'
        };
        
        notification.innerHTML = `
            <span class="notification-icon">${icons[type] || 'ℹ'}</span>
            <span class="notification-message">${message}</span>
            <div class="notification-progress"></div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
    
    /**
     * Show tooltip
     * @param {Event} e - Mouse event
     */
    function showTooltip(e) {
        const tooltip = document.createElement('div');
        tooltip.classList.add('cosmic-tooltip');
        tooltip.textContent = this.getAttribute('data-tooltip');
        
        // Position tooltip
        document.body.appendChild(tooltip);
        const rect = this.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width / 2 - tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
        
        // Add show class to trigger animation
        setTimeout(() => {
            tooltip.classList.add('show');
        }, 10);
        
        // Store reference to this tooltip
        this._tooltip = tooltip;
    }
    
    /**
     * Hide tooltip
     */
    function hideTooltip() {
        if (this._tooltip) {
            this._tooltip.classList.remove('show');
            setTimeout(() => {
                if (this._tooltip && this._tooltip.parentNode) {
                    this._tooltip.parentNode.removeChild(this._tooltip);
                }
                this._tooltip = null;
            }, 300);
        }
    }
    
    /**
     * Debounce function to limit how often a function is called
     * @param {Function} func - The function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} - Debounced function
     */
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }
});
