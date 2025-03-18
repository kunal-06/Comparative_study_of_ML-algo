// Log when the script loads
console.log("Script loaded successfully");

// Variables to store chart instances
let barChartInstance = null;
let pieChartInstance = null;
let lineChartInstance = null;
let radarChartInstance = null;
let doughnutChartInstance = null;
let polarAreaChartInstance = null;
let scatterChartInstance = null;

// Chart color scheme
const chartColors = {
    backgroundColors: [
        'rgba(99, 102, 241, 0.2)',
        'rgba(129, 140, 248, 0.2)',
        'rgba(56, 189, 248, 0.2)',
        'rgba(79, 70, 229, 0.2)',
        'rgba(192, 38, 211, 0.2)',
        'rgba(232, 121, 249, 0.2)',
        'rgba(168, 85, 247, 0.2)'
    ],
    borderColors: [
        'rgba(99, 102, 241, 1)',
        'rgba(129, 140, 248, 1)',
        'rgba(56, 189, 248, 1)',
        'rgba(79, 70, 229, 1)', 
        'rgba(192, 38, 211, 1)',
        'rgba(232, 121, 249, 1)',
        'rgba(168, 85, 247, 1)'
    ],
    solidColors: [
        '#6366f1',
        '#818cf8',
        '#38bdf8',
        '#4f46e5',
        '#c026d3',
        '#e879f9',
        '#a855f7'
    ]
};

// Function to show/hide specific chart containers
function showTab(chartId) {
    console.log("Showing tab:", chartId);
    document.querySelectorAll('.chart-container').forEach(container => {
        container.classList.remove('active');
    });
    document.getElementById(chartId).classList.add('active');

    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    document.querySelector(`button[onclick="showTab('${chartId}')"]`).classList.add('active');
}

// Create a table to display hyperparameter tuning results
function createHyperparameterTable(algorithmName, bestParams) {
    const table = document.createElement('table');
    table.className = 'hyperparameter-table';
    
    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const headerParams = document.createElement('th');
    headerParams.textContent = 'Parameter';
    const headerValue = document.createElement('th');
    headerValue.textContent = 'Best Value';
    headerRow.appendChild(headerParams);
    headerRow.appendChild(headerValue);
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Add rows for each parameter
    for (const [param, value] of Object.entries(bestParams)) {
        const row = document.createElement('tr');
        
        const paramCell = document.createElement('td');
        paramCell.textContent = param;
        
        const valueCell = document.createElement('td');
        valueCell.textContent = value;
        
        row.appendChild(paramCell);
        row.appendChild(valueCell);
        tbody.appendChild(row);
    }
    
    table.appendChild(tbody);
    return table;
}

// Create a chart section with title
function createChartSection(title, chartId, width = 'item-half') {
    const container = document.createElement('div');
    container.style.width = "100%";
    container.style.height = "350px";
    container.style.padding = "0";
    
    const header = document.createElement('div');
    header.className = 'dashboard-item-header';
    header.style.padding = "0 0.5rem";
    
    const titleElement = document.createElement('div');
    titleElement.className = 'dashboard-item-title';
    titleElement.innerHTML = `<i class="fas fa-chart-line"></i> ${title}`;
    
    header.appendChild(titleElement);
    container.appendChild(header);
    
    const canvasContainer = document.createElement('div');
    canvasContainer.style.width = "100%";
    canvasContainer.style.height = "310px";
    canvasContainer.style.position = "relative";
    canvasContainer.style.padding = "0.75rem";
    
    const canvas = document.createElement('canvas');
    canvas.id = chartId;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    
    canvasContainer.appendChild(canvas);
    container.appendChild(canvasContainer);
    
    return container;
}

// Create a metric card for summary statistics
function createMetricCard(label, value, icon, trend = null, trendValue = null) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    
    const iconElement = document.createElement('div');
    iconElement.innerHTML = `<i class="fas ${icon}"></i>`;
    iconElement.className = 'metric-icon';
    
    const valueElement = document.createElement('div');
    valueElement.className = 'metric-value';
    valueElement.textContent = value;
    
    const labelElement = document.createElement('div');
    labelElement.className = 'metric-label';
    labelElement.textContent = label;
    
    card.appendChild(iconElement);
    card.appendChild(valueElement);
    card.appendChild(labelElement);
    
    if (trend) {
        const trendElement = document.createElement('div');
        trendElement.className = `trend-indicator ${trend === 'up' ? 'trend-up' : 'trend-down'}`;
        trendElement.innerHTML = `
            <i class="fas fa-arrow-${trend}"></i>
            <span>${trendValue}</span>
        `;
        card.appendChild(trendElement);
    }
    
    return card;
}

// Calculate summary statistics from results
function calculateStats(results) {
    const values = Object.values(results);
    
    // Calculate mean
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    
    // Find max and min values and their corresponding algorithms
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    const maxAlgo = Object.keys(results).find(key => results[key] === maxVal);
    const minAlgo = Object.keys(results).find(key => results[key] === minVal);
    
    // Calculate standard deviation
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    return {
        mean,
        max: maxVal,
        min: minVal,
        maxAlgo,
        minAlgo,
        stdDev,
        count: values.length
    };
}

// Async function to handle form submission - Must be declared globally for onclick to work
async function handleAnalysis() {
    console.log("Analysis started!");
    
    // Get form element
    const form = document.getElementById('uploadForm');
    
    // Check if file is selected
    const fileInput = document.getElementById('dataset');
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a CSV file first.');
        return;
    }
    
    const formData = new FormData(form);
    const resultsDiv = document.getElementById('results');

    // Show loading indicator
    resultsDiv.innerHTML = `
        <div class="loading-container">
            <div class="spinner"></div>
            <p>Training models with hyperparameter tuning...</p>
            <p class="loading-subtext">This may take a few minutes depending on dataset size.</p>
        </div>
    `;

    try {
        console.log("Submitting form data...");
        // Send dataset to the backend
        const response = await fetch('https://comparative-study-of-ml-algo.onrender.com/', {
            method: 'POST',
            body: formData,
        });
        
        console.log("Response received:", response.status);
        
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Data received:", data);
        
        // Handle error response
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Extract results from the response
        const results = data.results || {};
        const problemType = data.problem_type || "Classification";
        const metricName = data.metric || "Accuracy";
        const bestParams = data.best_params || {};
        
        console.log("Results:", results);
        console.log("Best parameters:", bestParams);
        
        if (Object.keys(results).length === 0) {
            throw new Error("No results returned from the server.");
        }
        
        // Extract data for Chart.js
        const labels = Object.keys(results);
        const values = Object.values(results).map((val) => {
            // Convert to percentage for accuracy, keep as decimal for regression metrics
            return problemType === "Classification" ? (val * 100).toFixed(2) : val.toFixed(4);
        });
        
        // Create numeric values for charts
        const numericValues = values.map(val => parseFloat(val));
        
        // Create metric label based on problem type
        const metricLabel = `${metricName}${problemType === "Classification" ? " (%)" : ""}`;
        
        // Calculate summary statistics
        const stats = calculateStats(data.results);
        
        // Calculate formatted values for display
        const formattedMean = problemType === "Classification" 
            ? (stats.mean * 100).toFixed(2) + "%" 
            : stats.mean.toFixed(4);
        
        const formattedMax = problemType === "Classification" 
            ? (stats.max * 100).toFixed(2) + "%" 
            : stats.max.toFixed(4);
        
        const formattedMin = problemType === "Classification" 
            ? (stats.min * 100).toFixed(2) + "%" 
            : stats.min.toFixed(4);
        
        const formattedStdDev = problemType === "Classification" 
            ? (stats.stdDev * 100).toFixed(2) + "%" 
            : stats.stdDev.toFixed(4);
        
        // Clear the loading indicator
        resultsDiv.innerHTML = '';
        
        // Create dashboard container
        const dashboard = document.createElement('div');
        dashboard.className = 'dashboard';
        dashboard.style.width = "100%";
        
        // Add problem type header
        const headerDiv = document.createElement('div');
        headerDiv.className = 'dashboard-item item-full';
        headerDiv.innerHTML = `
            <h2><i class="fas fa-project-diagram"></i> Results Overview <span class="problem-type">${problemType} Problem</span></h2>
            <p class="description">Analysis of ${labels.length} machine learning algorithms using ${metricName} as the evaluation metric</p>
        `;
        dashboard.appendChild(headerDiv);
        
        // Create metrics summary section
        const metricsDiv = document.createElement('div');
        metricsDiv.className = 'metrics-summary';
        
        // Add metrics cards
        metricsDiv.appendChild(createMetricCard('Best Algorithm', stats.maxAlgo, 'fa-trophy', 'up', formattedMax));
        metricsDiv.appendChild(createMetricCard('Average Performance', formattedMean, 'fa-calculator'));
        metricsDiv.appendChild(createMetricCard('Standard Deviation', formattedStdDev, 'fa-chart-line'));
        metricsDiv.appendChild(createMetricCard('Number of Algorithms', stats.count, 'fa-cogs'));
        
        // Add metrics to the dashboard
        const metricsContainer = document.createElement('div');
        metricsContainer.className = 'dashboard-item item-full';
        metricsContainer.appendChild(metricsDiv);
        dashboard.appendChild(metricsContainer);
        
        // Common chart options
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            devicePixelRatio: 2,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12,
                            weight: '500'
                        },
                        boxWidth: 15,
                        padding: 15
                    }
                },
                title: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: {
                        family: "'Poppins', sans-serif",
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: "'Poppins', sans-serif",
                        size: 12
                    },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: true,
                    usePointStyle: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                if (problemType === "Classification") {
                                    label += (context.parsed.y * 100).toFixed(2) + '%';
                                } else {
                                    label += context.parsed.y.toFixed(4);
                                }
                            }
                            return label;
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        };
        
        // Create charts grid directly in the dashboard
        const chartsGridContainer = document.createElement('div');
        chartsGridContainer.className = 'dashboard-item item-full';
        chartsGridContainer.style.padding = '0';
        chartsGridContainer.style.background = 'transparent';
        chartsGridContainer.style.boxShadow = 'none';
        chartsGridContainer.style.marginBottom = '2rem';
        
        // Create 3 rows with 2 columns each for the charts
        // Row 1
        const row1 = document.createElement('div');
        row1.className = 'charts-row';
        
        // Bar Chart Section - row 1, column 1
        const barChartSection = createChartSection('Performance Comparison', 'barChart');
        barChartSection.style.margin = '0';
        barChartSection.className = 'charts-column';
        row1.appendChild(barChartSection);
        
        // Radar Chart Section - row 1, column 2
        const radarChartSection = createChartSection('Algorithm Comparison Radar', 'radarChart');
        radarChartSection.style.margin = '0';
        radarChartSection.className = 'charts-column';
        row1.appendChild(radarChartSection);
        
        // Row 2
        const row2 = document.createElement('div');
        row2.className = 'charts-row';
        
        // Doughnut Chart Section - row 2, column 1
        const doughnutChartSection = createChartSection('Performance Distribution', 'doughnutChart');
        doughnutChartSection.style.margin = '0';
        doughnutChartSection.className = 'charts-column';
        row2.appendChild(doughnutChartSection);
        
        // Line Chart Section - row 2, column 2
        const lineChartSection = createChartSection('Performance Trend', 'lineChart');
        lineChartSection.style.margin = '0';
        lineChartSection.className = 'charts-column';
        row2.appendChild(lineChartSection);
        
        // Row 3
        const row3 = document.createElement('div');
        row3.className = 'charts-row';
        
        // Polar Area Chart Section - row 3, column 1
        const polarAreaChartSection = createChartSection('Relative Strengths', 'polarAreaChart');
        polarAreaChartSection.style.margin = '0';
        polarAreaChartSection.className = 'charts-column';
        row3.appendChild(polarAreaChartSection);
        
        // Scatter Chart Section - row 3, column 2
        const scatterChartSection = createChartSection('Algorithm Ranking', 'scatterChart');
        scatterChartSection.style.margin = '0';
        scatterChartSection.className = 'charts-column';
        row3.appendChild(scatterChartSection);
        
        // Add all rows to the container
        chartsGridContainer.appendChild(row1);
        chartsGridContainer.appendChild(row2);
        chartsGridContainer.appendChild(row3);
        
        // Add charts grid to dashboard
        dashboard.appendChild(chartsGridContainer);
        
        // Add hyperparameter section
        const hyperparamSection = document.createElement('div');
        hyperparamSection.className = 'dashboard-item item-full';
        hyperparamSection.innerHTML = '<h3><i class="fas fa-sliders-h"></i> Hyperparameter Tuning Results</h3>';
        
        // Create tables container
        const tablesContainer = document.createElement('div');
        tablesContainer.className = 'tables-container';
        
        // Add tables for each algorithm
        for (const [algorithmName, params] of Object.entries(bestParams)) {
            const algorithmContainer = document.createElement('div');
            algorithmContainer.className = 'algorithm-container';
            
            const algorithmTitle = document.createElement('h4');
            algorithmTitle.innerHTML = `<i class="fas fa-cogs"></i> ${algorithmName}`;
            algorithmContainer.appendChild(algorithmTitle);
            
            // Create performance indicator
            const performanceValue = results[algorithmName];
            const performanceDisplay = document.createElement('p');
            performanceDisplay.className = 'performance-indicator';
            performanceDisplay.innerHTML = `${metricName}: <strong>${problemType === "Classification" ? 
                                          (performanceValue * 100).toFixed(2) + "%" : 
                                          performanceValue.toFixed(4)}</strong>`;
            algorithmContainer.appendChild(performanceDisplay);
            
            // Create and add the hyperparameter table
            const table = createHyperparameterTable(algorithmName, params);
            algorithmContainer.appendChild(table);
            
            tablesContainer.appendChild(algorithmContainer);
        }
        
        hyperparamSection.appendChild(tablesContainer);
        dashboard.appendChild(hyperparamSection);
        
        // Add dashboard to results div
        resultsDiv.appendChild(dashboard);
        
        // Initialize charts after they've been added to the DOM
        // Wait for a moment to ensure the DOM is updated
        setTimeout(() => {
            // Make sure canvas elements are fully rendered
            window.requestAnimationFrame(() => {
                // Destroy existing chart instances before creating new ones
                if (barChartInstance) barChartInstance.destroy();
                if (pieChartInstance) pieChartInstance.destroy();
                if (lineChartInstance) lineChartInstance.destroy();
                if (radarChartInstance) radarChartInstance.destroy();
                if (doughnutChartInstance) doughnutChartInstance.destroy();
                if (polarAreaChartInstance) polarAreaChartInstance.destroy();
                if (scatterChartInstance) scatterChartInstance.destroy();
                
                // Bar Chart
                const barCtx = document.getElementById('barChart').getContext('2d');
                barChartInstance = new Chart(barCtx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: metricLabel,
                            data: numericValues,
                            backgroundColor: chartColors.backgroundColors,
                            borderColor: chartColors.borderColors,
                            borderWidth: 1,
                        }]
                    },
                    options: {
                        ...chartOptions,
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                ticks: {
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                }
                            },
                            x: {
                                grid: {
                                    display: false
                                },
                                ticks: {
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                }
                            }
                        }
                    }
                });
                
                // Radar Chart
                const radarCtx = document.getElementById('radarChart').getContext('2d');
                radarChartInstance = new Chart(radarCtx, {
                    type: 'radar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: metricLabel,
                            data: numericValues,
                            backgroundColor: 'rgba(99, 102, 241, 0.2)',
                            borderColor: 'rgba(99, 102, 241, 1)',
                            pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgba(99, 102, 241, 1)',
                            borderWidth: 2,
                            pointRadius: 4
                        }]
                    },
                    options: {
                        ...chartOptions,
                        elements: {
                            line: {
                                tension: 0.1
                            }
                        },
                        scales: {
                            r: {
                                angleLines: {
                                    color: 'rgba(0, 0, 0, 0.2)',
                                    lineWidth: 1.5
                                },
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.1)',
                                    circular: true
                                },
                                pointLabels: {
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 12,
                                        weight: 'bold'
                                    },
                                    color: 'rgba(0, 0, 0, 0.8)'
                                },
                                ticks: {
                                    backdropColor: 'rgba(255, 255, 255, 0.75)',
                                    backdropPadding: 2,
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    },
                                    showLabelBackdrop: true,
                                    z: 1
                                },
                                min: 0
                            }
                        }
                    }
                });
                
                // Doughnut Chart
                const doughnutCtx = document.getElementById('doughnutChart').getContext('2d');
                doughnutChartInstance = new Chart(doughnutCtx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: numericValues,
                            backgroundColor: chartColors.solidColors,
                            borderWidth: 1,
                            borderColor: '#fff'
                        }]
                    },
                    options: {
                        ...chartOptions,
                        cutout: '60%',
                    }
                });
                
                // Line Chart
                const lineCtx = document.getElementById('lineChart').getContext('2d');
                lineChartInstance = new Chart(lineCtx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: metricLabel,
                            data: numericValues,
                            fill: false,
                            borderColor: chartColors.borderColors[0],
                            tension: 0.4,
                            pointBackgroundColor: chartColors.borderColors[0],
                            pointBorderColor: '#fff',
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }]
                    },
                    options: {
                        ...chartOptions,
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                ticks: {
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                }
                            },
                            x: {
                                grid: {
                                    display: false
                                },
                                ticks: {
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                }
                            }
                        }
                    }
                });
                
                // Polar Area Chart
                const polarAreaCtx = document.getElementById('polarAreaChart').getContext('2d');
                polarAreaChartInstance = new Chart(polarAreaCtx, {
                    type: 'polarArea',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: numericValues,
                            backgroundColor: chartColors.solidColors,
                            borderWidth: 1,
                            borderColor: '#fff'
                        }]
                    },
                    options: chartOptions
                });
                
                // Scatter Chart (algorithm ranking)
                const scatterCtx = document.getElementById('scatterChart').getContext('2d');
                
                // Create ranking data points
                const rankData = numericValues.map((value, index) => ({
                    x: index + 1,
                    y: value,
                    label: labels[index]
                }));
                
                // Sort by performance (descending)
                rankData.sort((a, b) => b.y - a.y);
                
                // Reassign x values (ranks)
                rankData.forEach((point, index) => {
                    point.x = index + 1;
                });
                
                scatterChartInstance = new Chart(scatterCtx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Algorithm Ranking',
                            data: rankData,
                            backgroundColor: chartColors.solidColors[0],
                            pointRadius: 8,
                            pointHoverRadius: 10
                        }]
                    },
                    options: {
                        ...chartOptions,
                        plugins: {
                            ...chartOptions.plugins,
                            tooltip: {
                                ...chartOptions.plugins.tooltip,
                                callbacks: {
                                    label: function(context) {
                                        const point = context.raw;
                                        return `Rank ${point.x}: ${point.label} (${point.y})`;
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                title: {
                                    display: true,
                                    text: metricLabel,
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                },
                                ticks: {
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                }
                            },
                            x: {
                                type: 'linear',
                                position: 'bottom',
                                title: {
                                    display: true,
                                    text: 'Rank (lower is better)',
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                },
                                ticks: {
                                    stepSize: 1,
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 10
                                    }
                                }
                            }
                        }
                    }
                });
            });
        }, 300); // Increased timeout to ensure DOM is ready
        
    } catch (error) {
        console.error("Error processing data:", error);
        resultsDiv.innerHTML = `<div class="error-message">
            <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
            <p>${error.message}</p>
            <p>Check the browser console for more details.</p>
        </div>`;
    }
}

// Attach the event listener when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        console.log("Submit button found, attaching event listener");
        submitBtn.addEventListener('click', handleAnalysis);
    } else {
        console.error("Submit button not found!");
    }
});
