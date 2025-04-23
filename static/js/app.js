/**
 * Cosmic Finance App JavaScript
 * Handles interactive elements and fixes button issues
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('App initialized');
    
    // Fix for tech tabs on home page
    const techTabs = document.querySelectorAll('.tech-tabs .tab');
    
    if (techTabs && techTabs.length > 0) {
        console.log('Found tech tabs:', techTabs.length);
        
        techTabs.forEach(tab => {
            // Make sure the tab is clickable
            tab.style.cursor = 'pointer';
            tab.style.pointerEvents = 'auto';
            
            // Add click handler
            tab.addEventListener('click', function() {
                console.log('Tab clicked:', this.getAttribute('data-tab'));
                
                // Remove active class from all tabs
                techTabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Hide all content panels
                const allContents = document.querySelectorAll('.tab-content');
                allContents.forEach(content => content.classList.remove('active'));
                
                // Show selected content
                const tabId = this.getAttribute('data-tab');
                const targetContent = document.getElementById(tabId + '-content');
                if (targetContent) {
                    targetContent.classList.add('active');
                } else {
                    console.error('Could not find content for tab:', tabId);
                }
            });
        });
    }
    
    // Fix for chart tabs
    const chartTabs = document.querySelectorAll('.chart-tab');
    if (chartTabs && chartTabs.length > 0) {
        chartTabs.forEach(tab => {
            tab.style.cursor = 'pointer';
            tab.style.pointerEvents = 'auto';
            
            tab.addEventListener('click', function() {
                chartTabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                const chartId = this.getAttribute('data-chart');
                document.querySelectorAll('.chart').forEach(chart => {
                    chart.classList.remove('active');
                });
                
                const targetChart = document.getElementById(chartId);
                if (targetChart) {
                    targetChart.classList.add('active');
                }
            });
        });
    }
    
    // Fix for Model Toggles (RNN and RL model buttons)
    const modelToggles = document.querySelectorAll('.neo-toggle input[type="checkbox"]');
    if (modelToggles && modelToggles.length > 0) {
        modelToggles.forEach(toggle => {
            // Make sure toggles and their parent elements are properly styled for clicking
            toggle.style.cursor = 'pointer';
            
            if (toggle.parentElement) {
                toggle.parentElement.style.cursor = 'pointer';
                
                // Add click handler to parent for better accessibility
                toggle.parentElement.addEventListener('click', function(e) {
                    // Don't toggle if clicking on the input itself (it will toggle automatically)
                    if (e.target !== toggle) {
                        toggle.checked = !toggle.checked;
                        // Trigger change event
                        const event = new Event('change');
                        toggle.dispatchEvent(event);
                    }
                });
            }
            
            // Add change handler
            toggle.addEventListener('change', function() {
                console.log(`Model ${this.value} is now ${this.checked ? 'enabled' : 'disabled'}`);
            });
        });
    }
    
    // Fix for Generate Prediction button - CRITICAL FIX
    const generateBtn = document.querySelector('#generate-prediction');
    const formSubmitBtn = document.querySelector('.predictor-form button[type="submit"]');
    
    // Handle main generate prediction button
    if (generateBtn) {
        console.log('Found generate prediction button');
        
        // Ensure the button is properly styled
        generateBtn.style.cursor = 'pointer';
        generateBtn.style.pointerEvents = 'auto';
        
        // Make sure the form submission works correctly
        const form = generateBtn.closest('form');
        if (form) {
            // Direct click handler for the button
            generateBtn.addEventListener('click', function(e) {
                console.log('Generate button clicked directly');
                
                // Visual feedback
                this.classList.add('loading');
                this.textContent = 'Processing...';
                
                // Submit the form
                if (this.type !== 'submit') {
                    e.preventDefault();
                    form.submit();
                }
            });
            
            // Also handle the form submit event
            form.addEventListener('submit', function(e) {
                console.log('Form submitted');
                if (generateBtn) {
                    generateBtn.classList.add('loading');
                    generateBtn.textContent = 'Processing...';
                }
            });
        } else {
            console.error('Generate button not in a form!');
        }
    }
    
    // Handle form submit button separately if different from generateBtn
    if (formSubmitBtn && formSubmitBtn !== generateBtn) {
        console.log('Found separate form submit button');
        
        formSubmitBtn.style.cursor = 'pointer';
        formSubmitBtn.style.pointerEvents = 'auto';
        
        formSubmitBtn.addEventListener('click', function(e) {
            console.log('Form submit button clicked');
            this.classList.add('loading');
            this.textContent = 'Processing...';
        });
    }
    
    // Update system time
    updateSystemTime();
});

// Simulate form submission for testing
function simulateFormSubmission() {
    // Hide placeholder, show predictions
    const placeholder = document.querySelector('.placeholder-container');
    const predictions = document.querySelector('.prediction-panel');
    
    if (placeholder) placeholder.style.display = 'none';
    if (predictions) predictions.style.display = 'block';
    
    // Simulate training animation
    simulateTraining();
    
    // Create/update chart
    setTimeout(function() {
        createPredictionChart();
    }, 1000);
}

function simulateTraining() {
    const progress = document.querySelector('.progress-bar');
    const epochCounter = document.querySelector('.training-stat:first-child .stat-value');
    const lossCounter = document.querySelector('.training-stat:nth-child(2) .stat-value');
    const timeCounter = document.querySelector('.training-stat:nth-child(3) .stat-value');
    
    if (!progress) return;
    
    let width = 0;
    let epoch = 0;
    const totalEpochs = 20;
    let startTime = new Date();
    
    const interval = setInterval(function() {
        width += 5;
        epoch = Math.floor((width / 100) * totalEpochs);
        const elapsed = ((new Date() - startTime) / 1000).toFixed(1);
        
        progress.style.width = width + '%';
        
        if (epochCounter) epochCounter.textContent = `${epoch}/${totalEpochs}`;
        if (lossCounter) lossCounter.textContent = (0.05 * (1 - width/100) + 0.01).toFixed(5);
        if (timeCounter) timeCounter.textContent = elapsed + 's';
        
        if (width >= 100) {
            clearInterval(interval);
            if (progress.classList) progress.classList.add('complete');
        }
    }, 150);
}

function createPredictionChart() {
    const chartElement = document.getElementById('price-plot');
    if (!chartElement || !window.Plotly) return;
    
    // Create sample data for visualization
    const dates = [];
    const historicalPrices = [];
    const today = new Date();
    
    // Generate historical dates
    for (let i = 60; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        dates.push(date.toISOString().split('T')[0]);
        
        // Create realistic price data with some randomness
        const basePrice = 195;
        const randomFactor = Math.random() * 10 - 5;
        const trendFactor = i * 0.05;
        historicalPrices.push(basePrice - trendFactor + randomFactor);
    }
    
    // Last historical price
    const lastPrice = historicalPrices[historicalPrices.length - 1];
    
    // Create prediction data for different models
    const futureDates = [];
    const lstmPredictions = [];
    const rnnPredictions = [];
    const rlPredictions = [];
    
    // Generate future dates
    for (let i = 1; i <= 30; i++) {
        const date = new Date(today);
        date.setDate(date.getDate() + i);
        futureDates.push(date.toISOString().split('T')[0]);
        
        // Different models predict different trajectories
        lstmPredictions.push(lastPrice * (1 + (0.053 * i / 30)));
        rnnPredictions.push(lastPrice * (1 + (0.045 * i / 30)));
        rlPredictions.push(lastPrice * (1 + (0.067 * i / 30)));
    }
    
    // Update UI with prediction prices
    updatePredictionCards(lastPrice, lstmPredictions, rnnPredictions, rlPredictions);
    
    // Create the plot
    const data = [
        {
            x: dates,
            y: historicalPrices,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical Prices',
            line: {
                color: '#a4b8c4',
                width: 2
            }
        },
        {
            x: futureDates,
            y: lstmPredictions,
            type: 'scatter',
            mode: 'lines',
            name: 'LSTM Prediction',
            line: {
                color: '#00e5ff',
                width: 2,
                dash: 'dashdot'
            }
        },
        {
            x: futureDates,
            y: rnnPredictions,
            type: 'scatter',
            mode: 'lines',
            name: 'RNN Prediction',
            line: {
                color: '#76ff03',
                width: 2,
                dash: 'dot'
            }
        },
        {
            x: futureDates,
            y: rlPredictions,
            type: 'scatter',
            mode: 'lines',
            name: 'RL Prediction',
            line: {
                color: '#ff1744',
                width: 2,
                dash: 'dash'
            }
        }
    ];
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(10, 10, 21, 0.7)',
        font: {
            family: 'Rajdhani, sans-serif',
            color: '#a4b8c4'
        },
        margin: {
            l: 50,
            r: 20,
            t: 30,
            b: 50
        },
        xaxis: {
            title: 'Date',
            showgrid: true,
            gridcolor: 'rgba(164, 184, 196, 0.1)',
            gridwidth: 1
        },
        yaxis: {
            title: 'Price ($)',
            showgrid: true,
            gridcolor: 'rgba(164, 184, 196, 0.1)',
            gridwidth: 1
        },
        legend: {
            orientation: 'h',
            x: 0.5,
            y: 1.1,
            xanchor: 'center'
        }
    };
    
    Plotly.newPlot(chartElement, data, layout, {responsive: true});
    
    // Add random noise to prediction lines to make them more realistic
    setTimeout(enhancePredictionLines, 500);
}

function updatePredictionCards(currentPrice, lstmPredictions, rnnPredictions, rlPredictions) {
    // Get last prediction for each model
    const lstmFinal = lstmPredictions[lstmPredictions.length - 1];
    const rnnFinal = rnnPredictions[rnnPredictions.length - 1];
    const rlFinal = rlPredictions[rlPredictions.length - 1];
    
    // Calculate percentage changes
    const lstmChange = ((lstmFinal - currentPrice) / currentPrice * 100).toFixed(2);
    const rnnChange = ((rnnFinal - currentPrice) / currentPrice * 100).toFixed(2);
    const rlChange = ((rlFinal - currentPrice) / currentPrice * 100).toFixed(2);
    
    // Update UI elements
    if (document.getElementById('lstm-price')) {
        document.getElementById('lstm-price').textContent = '$' + lstmFinal.toFixed(2);
        document.getElementById('lstm-change').textContent = `${lstmChange}% ${lstmChange >= 0 ? '▲' : '▼'}`;
        document.getElementById('lstm-change').className = `prediction-change ${lstmChange >= 0 ? 'positive' : 'negative'}`;
    }
    
    if (document.getElementById('rnn-price')) {
        document.getElementById('rnn-price').textContent = '$' + rnnFinal.toFixed(2);
        document.getElementById('rnn-change').textContent = `${rnnChange}% ${rnnChange >= 0 ? '▲' : '▼'}`;
        document.getElementById('rnn-change').className = `prediction-change ${rnnChange >= 0 ? 'positive' : 'negative'}`;
    }
    
    if (document.getElementById('rl-price')) {
        document.getElementById('rl-price').textContent = '$' + rlFinal.toFixed(2);
        document.getElementById('rl-change').textContent = `${rlChange}% ${rlChange >= 0 ? '▲' : '▼'}`;
        document.getElementById('rl-change').className = `prediction-change ${rlChange >= 0 ? 'positive' : 'negative'}`;
    }
}

function enhancePredictionLines() {
    const chart = document.getElementById('price-plot');
    if (!chart || !window.Plotly || !chart.data) return;
    
    try {
        // Get chart data
        const data = Plotly.d3.select('#price-plot').data()[0].data;
        
        // Process each prediction trace
        for (let i = 1; i < data.length; i++) {
            const trace = data[i];
            
            // Skip if it's not a prediction trace or already has enough points
            if (!trace.name.includes('Prediction') || trace.x.length > 30) continue;
            
            // Store original first and last points
            const firstX = trace.x[0];
            const firstY = trace.y[0];
            const lastX = trace.x[trace.x.length - 1];
            const lastY = trace.y[trace.y.length - 1];
            
            // Create new points with realistic market noise
            const newX = [];
            const newY = [];
            
            // Keep first point
            newX.push(firstX);
            newY.push(firstY);
            
            // Convert dates to timestamps for interpolation
            const startTime = new Date(firstX).getTime();
            const endTime = new Date(lastX).getTime();
            const change = lastY - firstY;
            
            // Add intermediate points with realistic noise
            for (let j = 1; j < 25; j++) {
                const ratio = j / 25;
                const currentTime = startTime + (endTime - startTime) * ratio;
                const currentDate = new Date(currentTime).toISOString().split('T')[0];
                
                // Base value follows the trend line
                const baseValue = firstY + change * ratio;
                
                // Add realistic noise (more volatile for longer predictions)
                const noise = (Math.random() - 0.5) * Math.abs(change) * 0.1 * ratio;
                
                newX.push(currentDate);
                newY.push(baseValue + noise);
            }
            
            // Add last point
            newX.push(lastX);
            newY.push(lastY);
            
            // Update trace
            trace.x = newX;
            trace.y = newY;
        }
        
        // Update plot
        Plotly.redraw(chart);
    } catch (e) {
        console.error("Error enhancing prediction lines:", e);
    }
}

function updateSystemTime() {
    const now = new Date();
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    };
    const timeString = now.toLocaleDateString('en-US', options);
    
    const timeElements = document.querySelectorAll('.system-time, #systemTime, #footerTime');
    timeElements.forEach(el => {
        if (el) el.textContent = timeString;
    });
    
    setTimeout(updateSystemTime, 60000);
}
