// Reference to the form element
const form = document.getElementById('uploadForm');

// Canvas references for all chart containers
const barChartCanvas = document.getElementById('barChart').getContext('2d');
const pieChartCanvas = document.getElementById('pieChart').getContext('2d');
const lineChartCanvas = document.getElementById('lineChart').getContext('2d');
const radarChartCanvas = document.getElementById('radarChart').getContext('2d');

// Variables to store chart instances
let barChartInstance;
let pieChartInstance;
let lineChartInstance;
let radarChartInstance;

// Function to show/hide specific chart containers
function showTab(chartId) {
    document.querySelectorAll('.chart-container').forEach(container => {
        container.classList.remove('active');
    });
    document.getElementById(chartId).classList.add('active');

    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    document.querySelector(`button[onclick="showTab('${chartId}')"]`).classList.add('active');
}

// Form submit event listener
form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent default form submission behavior

    const formData = new FormData(form);

    // Send dataset to the backend
    const response = await fetch('/', {
        method: 'POST',
        body: formData,
    });

    const results = await response.json();
    

    // Extract data for Chart.js
    const labels = Object.keys(results);
    const data = Object.values(results).map((acc) => (acc * 100).toFixed(2));
    console.log(data)
    // Destroy existing chart instances before creating new ones
    if (barChartInstance) barChartInstance.destroy();
    if (pieChartInstance) pieChartInstance.destroy();
    if (lineChartInstance) lineChartInstance.destroy();
    if (radarChartInstance) radarChartInstance.destroy();

    // Create bar chart
    barChartInstance = new Chart(barChartCanvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: data,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
            }],
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Bar Chart' } } },
    });

    // Create pie chart
    pieChartInstance = new Chart(pieChartCanvas, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF'],
            }],
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Pie Chart' } } },
    });

    // Create line chart
    lineChartInstance = new Chart(lineChartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: data,
                fill: false,
                borderColor: '#4BC0C0',
                tension: 0.1,
            }],
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Line Chart' } } },
    });

    // Create radar chart
    radarChartInstance = new Chart(radarChartCanvas, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: data,
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                borderColor: 'rgba(153, 102, 255, 1)',
            }],
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Radar Chart' } } },
    });
});
