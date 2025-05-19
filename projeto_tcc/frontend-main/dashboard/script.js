Chart.register(ChartDataLabels);

function navigateTo(page) {
    window.location.href = page;
}

async function carregarDashboard() {
    try {
        const resposta = await fetch('http://127.0.0.1:5000/dashboard-data');
        const data = await resposta.json();

        // --- Atualiza gráfico de pizza ---
        const pizzaCtx = document.getElementById('pizzaChart').getContext('2d');
        new Chart(pizzaCtx, {
            type: 'pie',
            data: {
                labels: data.sentimentos.labels,
                datasets: [{
                    data: data.sentimentos.data,
                    backgroundColor: ['#a61d16', '#807c7c', '#3b7d3c']
                }]
            },
            options: {
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: 18
                            }
                        }
                    },
                    datalabels: {
                        color: '#fff',
                        font: {
                            weight: 'bold',
                            size: 14
                        },
                        formatter: (value, context) => {
                            const total = context.chart.data.datasets[0].data.reduce((acc, val) => acc + val, 0);
                            const percentage = (value / total * 100).toFixed(1);
                            return `${percentage}%`;
                        }
                    }
                }
            },
            plugins: [ChartDataLabels]
        });

        // --- Atualiza gráfico de pizza ---
        const pizzaCtx2 = document.getElementById('pizzaChart2').getContext('2d');
        new Chart(pizzaCtx2, {
            type: 'pie',
            data: {
                labels: data.sentimentos_modelo.labels,
                datasets: [{
                    data: data.sentimentos_modelo.data,
                    backgroundColor: ['#a61d16', '#807c7c', '#3b7d3c']
                }]
            },
            options: {
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: 18
                            }
                        }
                    },
                    datalabels: {
                        color: '#fff',
                        font: {
                            weight: 'bold',
                            size: 14
                        },
                        formatter: (value, context) => {
                            const total = context.chart.data.datasets[0].data.reduce((acc, val) => acc + val, 0);
                            const percentage = (value / total * 100).toFixed(1);
                            return `${percentage}%`;
                        }
                    }
                }
            },
            plugins: [ChartDataLabels]
        });

        // --- Atualiza tabela ---
        const tabelaBody = document.querySelector('#registrosTable tbody');
        tabelaBody.innerHTML = "";
        data.registros.forEach(registro => {
            const row = `<tr>
                <td>${registro.texto}</td>
                <td>${registro.sentimento}</td>
                <td>${new Date(registro.data_criacao).toLocaleString()}</td>
            </tr>`;
            tabelaBody.innerHTML += row;
        });

        // --- Atualiza gráfico de barras ---
        const barCtx = document.getElementById('barChart').getContext('2d');
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: data.analisesPorDia.labels,
                datasets: [
                    {
                        label: 'Positivo',
                        data: data.analisesPorDia.positivo,
                        backgroundColor: '#3b7d3c'
                    },
                    {
                        label: 'Neutro',
                        data: data.analisesPorDia.neutro,
                        backgroundColor: '#807c7c'
                    },
                    {
                        label: 'Negativo',
                        data: data.analisesPorDia.negativo,
                        backgroundColor: '#a61d16'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            font: { size: 16 }
                        }
                    },
                    datalabels: {
                        anchor: 'end',
                        align: 'top',
                        font: {
                            size: 12,
                            weight: 'bold'
                        },
                        color: '#000',
                        formatter: (value) => value > 0 ? value : ''
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        ticks: {
                            font: { size: 14 }
                        }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        ticks: {
                            font: { size: 14 }
                        }
                    }
                }
            },
            plugins: [ChartDataLabels]
        });


    } catch (error) {
        console.error('Erro ao carregar dashboard:', error);
    }
}

window.onload = carregarDashboard;