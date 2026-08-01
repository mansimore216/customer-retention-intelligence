(function () {
    const palette = ['#4f7cff', '#7b5cff', '#2fb872', '#e8b93f', '#ef7f3a', '#e7495a'];

    function setKpi(id, text) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = text;
        el.classList.remove('skeleton');
    }

    function barChart(canvasId, labelToValue, options) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const labels = Object.keys(labelToValue);
        const values = labels.map((l) => labelToValue[l]);

        new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: labels.map((_, i) => palette[i % palette.length]),
                    borderRadius: 6,
                }],
            },
            options: Object.assign({
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#9aa7c2' }, grid: { display: false } },
                    y: {
                        ticks: { color: '#6b7794', callback: (v) => `${v}%` },
                        grid: { color: '#24304a' },
                        beginAtZero: true,
                    },
                },
            }, options || {}),
        });
    }

    async function loadAll() {
        try {
            const [statsRes, insightsRes] = await Promise.all([
                fetch('/api/dashboard-stats'),
                fetch('/api/insights'),
            ]);
            const stats = await statsRes.json();
            const insights = await insightsRes.json();

            if (!statsRes.ok) throw new Error(stats.detail || stats.error);
            if (!insightsRes.ok) throw new Error(insights.detail || insights.error);

            const churned = insights.churn_distribution.Churned;
            const retained = insights.churn_distribution.Retained;
            const total = churned + retained;

            setKpi('insTotalCustomers', total.toLocaleString());
            setKpi('insChurned', churned.toLocaleString());
            setKpi('insRetained', retained.toLocaleString());
            setKpi('insChurnRate', `${((churned / total) * 100).toFixed(1)}%`);

            const distCanvas = document.getElementById('churnDistributionChart');
            if (distCanvas) {
                new Chart(distCanvas.getContext('2d'), {
                    type: 'doughnut',
                    data: {
                        labels: ['Churned', 'Retained'],
                        datasets: [{
                            data: [churned, retained],
                            backgroundColor: ['#e7495a', '#2fb872'],
                            borderWidth: 0,
                        }],
                    },
                    options: {
                        plugins: { legend: { position: 'bottom', labels: { color: '#9aa7c2', usePointStyle: true } } },
                        cutout: '62%',
                    },
                });
            }

            barChart('contractChart', insights.churn_rate_by_contract);
            barChart('tenureChart', insights.churn_rate_by_tenure_group);
            barChart('chargesChart', insights.churn_rate_by_monthly_charges);
            barChart('internetChart', insights.churn_rate_by_internet_service);
            barChart('paymentChart', insights.churn_rate_by_payment_method);
        } catch (err) {
            const grid = document.getElementById('insightsKpiGrid');
            if (grid) {
                const banner = document.createElement('div');
                banner.className = 'error-state';
                banner.style.gridColumn = '1 / -1';
                banner.innerHTML = `<strong>Could not load insights</strong><span>${err.message}</span>`;
                grid.appendChild(banner);
            }
        }
    }

    loadAll();
})();
