(function () {
    const SECTIONS = [
        { title: 'Customer Information', fields: ['gender', 'SeniorCitizen', 'Partner', 'Dependents'] },
        { title: 'Account Information', fields: ['tenure', 'Contract', 'PaperlessBilling', 'PaymentMethod'] },
        {
            title: 'Services',
            fields: [
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            ],
        },
        { title: 'Billing Information', fields: ['MonthlyCharges', 'TotalCharges'] },
    ];

    const NUMBER_FIELD_CONFIG = {
        tenure: { min: 0, max: 72, step: 1, suffix: 'months' },
        MonthlyCharges: { min: 0, max: 500, step: 0.5, prefix: '$' },
        TotalCharges: { min: 0, max: 10000, step: 1, prefix: '$' },
    };

    let schema = null;
    let factorChart = null;

    function el(tag, attrs, children) {
        const node = document.createElement(tag);
        Object.entries(attrs || {}).forEach(([key, value]) => {
            if (key === 'class') node.className = value;
            else if (key === 'text') node.textContent = value;
            else node.setAttribute(key, value);
        });
        (children || []).forEach((child) => node.appendChild(child));
        return node;
    }

    // ------------------------------------------------------------
    // Form construction
    // ------------------------------------------------------------

    function buildToggleField(field, options, defaultValue) {
        const wrap = el('div', { class: 'form-field' });
        wrap.appendChild(el('label', { class: 'form-label', text: schema.labels[field] || field }));

        const group = el('div', { class: 'toggle-group', 'data-field': field });
        options.forEach((option) => {
            const btn = el('button', {
                type: 'button',
                class: 'toggle-option' + (option === defaultValue ? ' active' : ''),
                'data-value': option,
                text: option,
            });
            btn.addEventListener('click', () => {
                group.querySelectorAll('.toggle-option').forEach((b) => b.classList.remove('active'));
                btn.classList.add('active');
                group.dataset.value = option;
            });
            group.appendChild(btn);
        });
        group.dataset.value = defaultValue;
        wrap.appendChild(group);
        return wrap;
    }

    function buildSelectField(field, options, defaultValue) {
        const wrap = el('div', { class: 'form-field' });
        wrap.appendChild(el('label', { class: 'form-label', for: `field-${field}`, text: schema.labels[field] || field }));

        const select = el('select', { class: 'form-select', id: `field-${field}`, 'data-field': field });
        options.forEach((option) => {
            const opt = el('option', { value: option, text: option });
            if (option === defaultValue) opt.selected = true;
            select.appendChild(opt);
        });
        wrap.appendChild(select);
        return wrap;
    }

    function buildNumberField(field, defaultValue) {
        const config = NUMBER_FIELD_CONFIG[field] || {};
        const wrap = el('div', { class: 'form-field' });
        wrap.appendChild(el('label', { class: 'form-label', for: `field-${field}`, text: schema.labels[field] || field }));

        const inputWrap = el('div', { class: 'number-input-wrap' });
        if (config.prefix) inputWrap.appendChild(el('span', { class: 'input-affix', text: config.prefix }));

        const input = el('input', {
            type: 'number',
            id: `field-${field}`,
            class: 'form-number',
            'data-field': field,
            value: defaultValue,
            min: config.min !== undefined ? config.min : '',
            max: config.max !== undefined ? config.max : '',
            step: config.step !== undefined ? config.step : 'any',
        });
        inputWrap.appendChild(input);
        if (config.suffix) inputWrap.appendChild(el('span', { class: 'input-affix', text: config.suffix }));

        wrap.appendChild(inputWrap);
        wrap.appendChild(el('span', { class: 'field-hint', id: `hint-${field}` }));
        return wrap;
    }

    function buildField(field) {
        const defaultValue = schema.defaults[field];
        const options = schema.options[field];

        if (field === 'SeniorCitizen') {
            const senior = schema.defaults.SeniorCitizen === 1 ? 'Yes' : 'No';
            return buildToggleField(field, ['No', 'Yes'], senior);
        }

        if (NUMBER_FIELD_CONFIG[field]) {
            return buildNumberField(field, defaultValue);
        }

        if (options && options.length <= 2) {
            return buildToggleField(field, options, defaultValue);
        }

        if (options) {
            return buildSelectField(field, options, defaultValue);
        }

        return buildNumberField(field, defaultValue);
    }

    function renderForm() {
        const container = document.getElementById('formSections');
        container.innerHTML = '';

        SECTIONS.forEach((section) => {
            const sectionEl = el('fieldset', { class: 'form-section' });
            sectionEl.appendChild(el('legend', { text: section.title }));

            const grid = el('div', { class: 'form-grid' });
            section.fields.forEach((field) => grid.appendChild(buildField(field)));
            sectionEl.appendChild(grid);

            container.appendChild(sectionEl);
        });

        wireDerivedFieldValidation();
    }

    // Keep dependent fields sane: PhoneService=No implies MultipleLines has
    // no meaningful "Yes"/"No" choice, InternetService=No implies the
    // internet add-ons don't apply. We nudge, not force, so power users can
    // still explore edge cases.
    function wireDerivedFieldValidation() {
        const phoneToggle = document.querySelector('.toggle-group[data-field="PhoneService"]');
        const linesSelect = document.querySelector('select[data-field="MultipleLines"]');
        const internetSelect = document.querySelector('select[data-field="InternetService"]');
        const addonFields = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'];

        function syncPhone() {
            if (!phoneToggle || !linesSelect) return;
            if (phoneToggle.dataset.value === 'No') {
                linesSelect.value = 'No phone service';
            } else if (linesSelect.value === 'No phone service') {
                linesSelect.value = 'No';
            }
        }

        function syncInternet() {
            if (!internetSelect) return;
            const noInternet = internetSelect.value === 'No';
            addonFields.forEach((field) => {
                const select = document.querySelector(`select[data-field="${field}"]`);
                if (!select) return;
                if (noInternet) select.value = 'No internet service';
                else if (select.value === 'No internet service') select.value = 'No';
            });
        }

        if (phoneToggle) {
            phoneToggle.querySelectorAll('.toggle-option').forEach((btn) => btn.addEventListener('click', syncPhone));
        }
        if (internetSelect) internetSelect.addEventListener('change', syncInternet);
    }

    function collectFormValues() {
        const values = {};
        schema.field_order.forEach((field) => {
            const toggle = document.querySelector(`.toggle-group[data-field="${field}"]`);
            const select = document.querySelector(`select[data-field="${field}"]`);
            const number = document.querySelector(`input[data-field="${field}"]`);

            if (field === 'SeniorCitizen' && toggle) {
                values[field] = toggle.dataset.value === 'Yes' ? 1 : 0;
            } else if (toggle) {
                values[field] = toggle.dataset.value;
            } else if (select) {
                values[field] = select.value;
            } else if (number) {
                values[field] = number.value === '' ? null : Number(number.value);
            }
        });
        return values;
    }

    // ------------------------------------------------------------
    // Results rendering
    // ------------------------------------------------------------

    const RISK_META = {
        Low: { label: 'Low Churn Risk', className: 'risk-low', color: '#2fb872' },
        Medium: { label: 'Medium Churn Risk', className: 'risk-medium', color: '#e8b93f' },
        High: { label: 'High Churn Risk', className: 'risk-high', color: '#ef7f3a' },
    };

    function renderResult(result) {
        const container = document.getElementById('predictResults');
        container.innerHTML = '';

        const meta = RISK_META[result.risk_level] || RISK_META.Medium;
        const probabilityPct = (result.churn_probability * 100).toFixed(1);

        const resultCard = el('div', { class: `result-hero ${meta.className}` });
        resultCard.appendChild(el('div', { class: 'result-risk-label', text: meta.label }));
        resultCard.appendChild(el('div', { class: 'result-probability', text: `${probabilityPct}%` }));
        resultCard.appendChild(el('div', { class: 'result-probability-caption', text: 'predicted probability of churn' }));

        const track = el('div', { class: 'progress-track' });
        const fill = el('div', { class: 'progress-fill' });
        fill.style.width = `${probabilityPct}%`;
        fill.style.background = meta.color;
        track.appendChild(fill);
        resultCard.appendChild(track);

        const predictionRow = el('div', { class: 'metric-row' });
        predictionRow.appendChild(el('span', { class: 'metric-label', text: 'Predicted Churn (Yes/No)' }));
        predictionRow.appendChild(el('span', { class: 'metric-value', text: result.prediction }));
        resultCard.appendChild(predictionRow);

        container.appendChild(resultCard);

        const explanationCard = el('div', { class: 'panel-card' });
        explanationCard.appendChild(el('h3', { text: 'Why this prediction' }));
        explanationCard.appendChild(el('p', { class: 'driver-text', text: result.explanation }));
        container.appendChild(explanationCard);

        const explainCard = el('div', { class: 'panel-card' });
        explainCard.appendChild(el('h3', { text: 'Top Factors (SHAP Explainability)' }));
        const chartBody = el('div', { class: 'chart-body factor-chart-body' });
        const canvas = el('canvas', { id: 'factorChart' });
        chartBody.appendChild(canvas);
        explainCard.appendChild(chartBody);
        container.appendChild(explainCard);

        const recCard = el('div', { class: 'panel-card' });
        recCard.appendChild(el('h3', { text: 'Recommended Retention Actions' }));
        const recList = el('div', { class: 'recommendation-list' });
        result.recommendations.forEach((text, idx) => {
            recList.appendChild(el('div', { class: 'recommendation-card' }, [
                el('span', { class: 'recommendation-index', text: String(idx + 1) }),
                el('span', { class: 'recommendation-text', text }),
            ]));
        });
        recCard.appendChild(recList);
        container.appendChild(recCard);

        renderFactorChart(result.top_factors);
    }

    function renderFactorChart(topFactors) {
        const canvas = document.getElementById('factorChart');
        if (!canvas) return;

        if (factorChart) {
            factorChart.destroy();
            factorChart = null;
        }

        const ordered = [...topFactors].reverse();
        const labels = ordered.map((f) => `${f.label} (${f.value})`);
        const values = ordered.map((f) => (f.direction === 'increases_risk' ? f.relative_strength : -f.relative_strength));
        const colors = ordered.map((f) => (f.direction === 'increases_risk' ? '#ef7f3a' : '#2fb872'));

        factorChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 4,
                    barThickness: 16,
                }],
            },
            options: {
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => (ctx.raw >= 0 ? 'Increases churn risk' : 'Decreases churn risk'),
                        },
                    },
                },
                scales: {
                    x: {
                        min: -1,
                        max: 1,
                        ticks: { color: '#6b7794', callback: () => '' },
                        grid: { color: '#24304a' },
                    },
                    y: {
                        ticks: { color: '#9aa7c2', font: { size: 11.5 } },
                        grid: { display: false },
                    },
                },
            },
        });
    }

    function showError(message) {
        const errorEl = document.getElementById('formError');
        errorEl.textContent = message;
    }

    function setSubmitting(isSubmitting) {
        const button = document.getElementById('predictButton');
        button.classList.toggle('is-loading', isSubmitting);
        button.disabled = isSubmitting;
    }

    async function handleSubmit(evt) {
        evt.preventDefault();
        showError('');
        setSubmitting(true);

        try {
            const payload = collectFormValues();
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.error || 'Prediction failed.');
            renderResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            setSubmitting(false);
        }
    }

    async function init() {
        try {
            const res = await fetch('/api/form-schema');
            schema = await res.json();
            renderForm();
        } catch (err) {
            document.getElementById('formSections').innerHTML =
                `<div class="error-state"><strong>Could not load form</strong><span>${err.message}</span></div>`;
            return;
        }

        document.getElementById('predictForm').addEventListener('submit', handleSubmit);
    }

    init();
})();
