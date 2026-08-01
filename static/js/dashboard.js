(function () {
    const fmtPercent = (value) => `${(value * 100).toFixed(1)}%`;
    const fmtNumber = (value) => Number(value).toLocaleString();

    function setKpi(id, text) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = text;
        el.classList.remove('skeleton');
    }

    function showError(message) {
        const grid = document.getElementById('kpiGrid');
        if (!grid) return;
        const banner = document.createElement('div');
        banner.className = 'error-state';
        banner.style.gridColumn = '1 / -1';
        banner.innerHTML = `<strong>Could not load dashboard data</strong><span>${message}</span>`;
        grid.appendChild(banner);
    }

    async function loadDashboard() {
        try {
            const res = await fetch('/api/dashboard-stats');
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.error || 'Unknown error');

            setKpi('kpiTotalCustomers', fmtNumber(data.total_customers));
            setKpi('kpiHighRisk', fmtNumber(data.high_risk_customers));
            setKpi('kpiAvgChurn', fmtPercent(data.average_churn_probability));
            setKpi('kpiRetentionOpportunities', fmtNumber(data.retention_opportunities));
        } catch (err) {
            showError(err.message);
        }
    }

    async function loadRiskChart() {
        const canvas = document.getElementById('riskDistributionChart');
        if (!canvas) return;

        try {
            const res = await fetch('/api/risk-distribution');
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.error || 'Unknown error');

            const labels = ['Low', 'Medium', 'High'];
            const colors = ['#2fb872', '#e8b93f', '#ef7f3a'];

            new Chart(canvas.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels,
                    datasets: [{
                        data: labels.map((l) => data[l] || 0),
                        backgroundColor: colors,
                        borderWidth: 0,
                    }],
                },
                options: {
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#9aa7c2', usePointStyle: true } },
                    },
                    cutout: '62%',
                },
            });
        } catch (err) {
            canvas.parentElement.innerHTML = `<div class="error-state"><strong>Chart failed to load</strong><span>${err.message}</span></div>`;
        }
    }

    loadDashboard();
    loadRiskChart();
})();
