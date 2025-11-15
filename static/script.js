let file1 = null;
let file2 = null;
let chart = null;

// File input handlers
document.getElementById('file1').addEventListener('change', function(e) {
    file1 = e.target.files[0];
    document.getElementById('file1-name').textContent = file1 ? file1.name : 'No file selected';
    updateAnalyzeButton();
});

document.getElementById('file2').addEventListener('change', function(e) {
    file2 = e.target.files[0];
    document.getElementById('file2-name').textContent = file2 ? file2.name : 'No file selected';
    updateAnalyzeButton();
});

function updateAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = !(file1 && file2);
}

// Analyze button click handler
document.getElementById('analyzeBtn').addEventListener('click', async function() {
    if (!file1 || !file2) {
        showError('Please select both documents');
        return;
    }

    // Hide previous results and errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('loading').style.display = 'block';

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function displayResults(data) {
    // Display similarity metrics
    const metrics = data.similarity_metrics;
    
    updateMetric('cosine', metrics.cosine_similarity);
    updateMetric('jaccard', metrics.jaccard_index);
    updateMetric('lsa', metrics.lsa_similarity);
    updateMetric('bert', metrics.bert_similarity);

    // Display IR metrics
    const irMetrics = data.ir_evaluation_metrics;
    document.getElementById('precision-value').textContent = irMetrics.precision.toFixed(4);
    document.getElementById('recall-value').textContent = irMetrics.recall.toFixed(4);
    document.getElementById('f1-value').textContent = irMetrics.f1_score.toFixed(4);

    // Display document info
    const docInfo = data.document_info;
    document.getElementById('doc1-length').textContent = docInfo.doc1_length.toLocaleString();
    document.getElementById('doc1-words').textContent = docInfo.doc1_words.toLocaleString();
    document.getElementById('doc2-length').textContent = docInfo.doc2_length.toLocaleString();
    document.getElementById('doc2-words').textContent = docInfo.doc2_words.toLocaleString();

    // Display average similarity
    document.getElementById('avg-score').textContent = data.average_similarity.toFixed(4);

    // Display embeddings
    displayEmbeddings(data.embeddings);

    // Create chart
    createChart(metrics);

    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Smooth scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayEmbeddings(embeddings) {
    // Display TF-IDF vectors for Document 1
    const tfidfDoc1Html = createTfidfTable(embeddings.tfidf_doc1, "Document 1");
    document.getElementById('tfidf-doc1').innerHTML = tfidfDoc1Html;

    // Display TF-IDF vectors for Document 2
    const tfidfDoc2Html = createTfidfTable(embeddings.tfidf_doc2, "Document 2");
    document.getElementById('tfidf-doc2').innerHTML = tfidfDoc2Html;

    // Display BERT embeddings
    const bertHtml = createBertDisplay(embeddings);
    document.getElementById('bert-embeddings').innerHTML = bertHtml;
}

function createTfidfTable(features, docName) {
    if (!features || features.length === 0) {
        return '<p class="loading-text">No features available</p>';
    }

    const maxScore = Math.max(...features.map(f => f.score));
    
    let html = `
        <div class="embedding-info">
            <p><strong>Top ${features.length} TF-IDF Features</strong></p>
            <p>These features have the highest importance scores in ${docName}</p>
        </div>
        <table class="embedding-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>TF-IDF Score</th>
                    <th>Weight</th>
                </tr>
            </thead>
            <tbody>
    `;

    features.forEach(feature => {
        const barWidth = (feature.score / maxScore) * 100;
        html += `
            <tr>
                <td class="feature-name">${feature.feature}</td>
                <td class="feature-score">${feature.score.toFixed(4)}</td>
                <td>
                    <div class="feature-bar-container">
                        <div class="feature-bar" style="width: ${barWidth}%"></div>
                    </div>
                </td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    return html;
}

function createBertDisplay(embeddings) {
    let html = `
        <div class="embedding-info">
            <p><strong>BERT Embedding Dimensions:</strong> ${embeddings.bert_dimensions}</p>
            <p>Showing first 10 dimensions as a sample (full embeddings are ${embeddings.bert_dimensions}D)</p>
        </div>
        
        <h4 style="color: var(--text-primary); margin: 1.5rem 0 1rem 0;">Document 1 Sample</h4>
        <div class="bert-sample">
    `;

    embeddings.bert_doc1_sample.forEach((value, index) => {
        html += `
            <div class="bert-dimension">
                <div class="dim-label">Dim ${index}</div>
                <div class="dim-value">${value.toFixed(4)}</div>
            </div>
        `;
    });

    html += `
        </div>
        
        <h4 style="color: var(--text-primary); margin: 1.5rem 0 1rem 0;">Document 2 Sample</h4>
        <div class="bert-sample">
    `;

    embeddings.bert_doc2_sample.forEach((value, index) => {
        html += `
            <div class="bert-dimension">
                <div class="dim-label">Dim ${index}</div>
                <div class="dim-value">${value.toFixed(4)}</div>
            </div>
        `;
    });

    html += '</div>';

    return html;
}

function updateMetric(name, value) {
    const percentage = (value * 100).toFixed(2);
    document.getElementById(`${name}-value`).textContent = value.toFixed(4);
    document.getElementById(`${name}-bar`).style.width = `${percentage}%`;
}

function createChart(metrics) {
    const ctx = document.getElementById('similarityChart').getContext('2d');

    // Destroy previous chart if exists
    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Cosine Similarity', 'Jaccard Index', 'LSA Similarity', 'BERT Similarity'],
            datasets: [{
                label: 'Similarity Score',
                data: [
                    metrics.cosine_similarity,
                    metrics.jaccard_index,
                    metrics.lsa_similarity,
                    metrics.bert_similarity
                ],
                backgroundColor: [
                    'rgba(88, 166, 255, 0.8)',
                    'rgba(63, 185, 80, 0.8)',
                    'rgba(187, 128, 255, 0.8)',
                    'rgba(255, 128, 128, 0.8)'
                ],
                borderColor: [
                    'rgba(88, 166, 255, 1)',
                    'rgba(63, 185, 80, 1)',
                    'rgba(187, 128, 255, 1)',
                    'rgba(255, 128, 128, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(22, 27, 34, 0.95)',
                    titleColor: '#e6edf3',
                    bodyColor: '#e6edf3',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return 'Score: ' + context.parsed.y.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        color: '#8b949e',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: '#30363d',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        color: '#8b949e',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}
