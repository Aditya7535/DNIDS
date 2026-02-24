/* =============================================================
   DNIDS Dashboard - Main JavaScript
   ============================================================= */

// ─── State ───
let pollInterval = null;
let timelineChart = null;
let distributionChart = null;
let isSimulationRunning = false;

// ─── Elements ───
const slider = document.getElementById('recordCount');
const valueDisplay = document.getElementById('recordValue');
const btnSimulate = document.getElementById('btnSimulate');
const progressContainer = document.getElementById('progressContainer');
const statsSection = document.getElementById('statsSection');
const chartsSection = document.getElementById('chartsSection');
const detailsSection = document.getElementById('detailsSection');

// ─── Slider ───
slider.addEventListener('input', () => {
    valueDisplay.textContent = Number(slider.value).toLocaleString();
});

// ─── Format Numbers ───
function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
}

// ─── Start Simulation ───
async function startSimulation() {
    if (isSimulationRunning) return;

    const records = parseInt(slider.value);

    // UI updates
    btnSimulate.disabled = true;
    btnSimulate.innerHTML = `
        <svg class="spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
        </svg>
        Initializing...`;
    progressContainer.style.display = 'block';
    statsSection.style.display = 'block';
    chartsSection.style.display = 'block';
    detailsSection.style.display = 'block';

    // Activate pipeline nodes
    document.querySelectorAll('.pipeline-node').forEach(n => n.classList.add('active'));

    // Update status
    setStatus('running', 'Running');

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ records })
        });

        if (!res.ok) {
            const err = await res.json();
            alert(err.error || 'Failed to start simulation');
            resetUI();
            return;
        }

        isSimulationRunning = true;
        btnSimulate.innerHTML = `
            <svg class="spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
            </svg>
            Processing...`;

        // Start polling
        pollInterval = setInterval(pollStats, 500);

    } catch (e) {
        console.error('Error:', e);
        alert('Failed to connect to server. Please make sure the app is running.');
        resetUI();
    }
}

// ─── Poll Statistics ───
async function pollStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        updateDashboard(data);

        if (!data.running && isSimulationRunning) {
            // Simulation finished
            clearInterval(pollInterval);
            isSimulationRunning = false;

            setStatus('active', 'Complete');
            document.querySelectorAll('.pipeline-node').forEach(n => n.classList.remove('active'));

            btnSimulate.disabled = false;
            btnSimulate.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Run Again`;
        }
    } catch (e) {
        console.error('Poll error:', e);
    }
}

// ─── Update Dashboard ───
function updateDashboard(data) {
    // Progress
    document.getElementById('progressFill').style.width = data.progress + '%';
    document.getElementById('progressPercent').textContent = data.progress + '%';
    document.getElementById('progressLabel').textContent =
        data.running ? 'Processing network traffic...' : 'Processing complete!';
    document.getElementById('progressRecords').textContent =
        `${formatNumber(data.records_processed)} / ${formatNumber(data.total_records)} records`;
    document.getElementById('progressRate').textContent =
        `${formatNumber(data.rate)} records/sec`;

    // Stats cards
    document.getElementById('statTotalValue').textContent = formatNumber(data.records_processed);
    document.getElementById('statAttackValue').textContent = formatNumber(data.attacks_detected);
    document.getElementById('statNormalValue').textContent = formatNumber(data.normals_detected);
    document.getElementById('statAccuracyValue').textContent = data.accuracy + '%';
    document.getElementById('statPrecisionValue').textContent = data.precision + '%';
    document.getElementById('statRecallValue').textContent = data.recall + '%';

    // Confusion matrix
    document.getElementById('cmTP').textContent = formatNumber(data.true_positives);
    document.getElementById('cmTN').textContent = formatNumber(data.true_negatives);
    document.getElementById('cmFP').textContent = formatNumber(data.false_positives);
    document.getElementById('cmFN').textContent = formatNumber(data.false_negatives);

    // Charts
    updateTimelineChart(data.timeline);
    updateDistributionChart(data.attacks_detected, data.normals_detected);

    // Attack types
    updateAttackTypes(data.attack_types);

    // Alerts
    updateAlerts(data.recent_alerts);
}

// ─── Timeline Chart ───
function updateTimelineChart(timeline) {
    const ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    const labels = timeline.map(t => t.record);
    const attacks = timeline.map(t => t.attacks);
    const normals = timeline.map(t => t.normals);
    const accuracy = timeline.map(t => t.accuracy);

    if (timelineChart) {
        timelineChart.data.labels = labels;
        timelineChart.data.datasets[0].data = attacks;
        timelineChart.data.datasets[1].data = normals;
        timelineChart.data.datasets[2].data = accuracy;
        timelineChart.update('none');
        return;
    }

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Attacks',
                    data: attacks,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y',
                },
                {
                    label: 'Normal',
                    data: normals,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y',
                },
                {
                    label: 'Accuracy %',
                    data: accuracy,
                    borderColor: '#7c3aed',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: "'Inter'", size: 11 }, usePointStyle: true, pointStyleWidth: 12, boxHeight: 6 }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(148, 163, 184, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    bodyFont: { family: "'JetBrains Mono'" },
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'Record #', color: '#64748b', font: { size: 11 } }
                },
                y: {
                    position: 'left',
                    grid: { color: 'rgba(148, 163, 184, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'Count', color: '#64748b', font: { size: 11 } }
                },
                y1: {
                    position: 'right',
                    min: 0,
                    max: 100,
                    grid: { display: false },
                    ticks: { color: '#7c3aed', font: { size: 10 }, callback: v => v + '%' },
                    title: { display: true, text: 'Accuracy', color: '#7c3aed', font: { size: 11 } }
                }
            }
        }
    });

    // Set canvas height
    ctx.parentElement.style.height = '250px';
}

// ─── Distribution Chart ───
function updateDistributionChart(attacks, normals) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    if (distributionChart) {
        distributionChart.data.datasets[0].data = [normals, attacks];
        distributionChart.update('none');
        return;
    }

    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal Traffic', 'Attacks Detected'],
            datasets: [{
                data: [normals, attacks],
                backgroundColor: ['rgba(16, 185, 129, 0.8)', 'rgba(239, 68, 68, 0.8)'],
                borderColor: ['rgba(16, 185, 129, 1)', 'rgba(239, 68, 68, 1)'],
                borderWidth: 2,
                hoverBorderWidth: 3,
                spacing: 4,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { family: "'Inter'", size: 11 },
                        usePointStyle: true,
                        pointStyleWidth: 12,
                        padding: 16,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(148, 163, 184, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    bodyFont: { family: "'JetBrains Mono'" },
                }
            }
        }
    });

    ctx.parentElement.style.height = '250px';
}

// ─── Attack Types ───
function updateAttackTypes(attackTypes) {
    const container = document.getElementById('attackTypesList');
    const countEl = document.getElementById('attackTypeCount');

    const entries = Object.entries(attackTypes).sort((a, b) => b[1] - a[1]);
    countEl.textContent = entries.length + ' types';

    if (entries.length === 0) {
        container.innerHTML = '<div class="empty-state">No attacks detected yet</div>';
        return;
    }

    const maxCount = entries[0][1];
    container.innerHTML = entries.map(([name, count]) => {
        const pct = (count / maxCount * 100).toFixed(0);
        return `
            <div class="attack-type-item">
                <span class="attack-type-name">${escapeHtml(name)}</span>
                <div class="attack-type-bar">
                    <div class="attack-type-bar-fill" style="width:${pct}%"></div>
                </div>
                <span class="attack-type-count">${formatNumber(count)}</span>
            </div>`;
    }).join('');
}

// ─── Alerts ───
function updateAlerts(alerts) {
    const container = document.getElementById('alertsList');
    const countEl = document.getElementById('alertCount');
    countEl.textContent = alerts.length + ' alerts';

    if (alerts.length === 0) {
        container.innerHTML = '<div class="empty-state">No alerts yet</div>';
        return;
    }

    container.innerHTML = alerts.slice(0, 30).map(a => `
        <div class="alert-item ${a.correct ? 'correct' : 'incorrect'}">
            <div class="alert-indicator"></div>
            <div class="alert-info">
                <div class="alert-type">${escapeHtml(a.attack_type)}</div>
                <div class="alert-meta">Record #${a.record_id} • ${a.timestamp}</div>
            </div>
            <div class="alert-confidence">${(a.confidence * 100).toFixed(1)}%</div>
        </div>`
    ).join('');
}

// ─── Status ───
function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    dot.className = 'status-dot ' + state;
    txt.textContent = text;
}

// ─── Reset UI ───
function resetUI() {
    btnSimulate.disabled = false;
    btnSimulate.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Run Detection`;
    setStatus('', 'Idle');
    document.querySelectorAll('.pipeline-node').forEach(n => n.classList.remove('active'));
}

// ─── Utility ───
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ─── Spin Animation (CSS injection) ───
const style = document.createElement('style');
style.textContent = `
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .spin { animation: spin 1s linear infinite; }
`;
document.head.appendChild(style);

// ─── Initialize ───
document.addEventListener('DOMContentLoaded', () => {
    setStatus('', 'Ready');

    // Smooth entrance for architecture cards
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.arch-card').forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = `opacity 0.5s ease ${i * 0.1}s, transform 0.5s ease ${i * 0.1}s`;
        observer.observe(card);
    });
});
