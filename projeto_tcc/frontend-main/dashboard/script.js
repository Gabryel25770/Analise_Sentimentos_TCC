Chart.register(ChartDataLabels);

function navigateTo(page) {
    window.location.href = page;
    
}

let pizzaChart1 = null;
let pizzaChart2 = null;
let barChart = null;

let primeiraAtualizacao = true;

const DEFAULT_DELAY_REFRESH_GRAP = 10000; //10 sec
window.setInterval(() => {
    carregarDashboard();
 
}, DEFAULT_DELAY_REFRESH_GRAP);
 


async function carregarDashboard() {
    try {
        const resposta = await fetch('https://api.analisefeedback.com.br/dashboard-data');
        const data = await resposta.json();

        // --- Atualiza gráfico de pizza ---
        if (pizzaChart1) pizzaChart1.destroy();
        const pizzaCtx = document.getElementById('pizzaChart').getContext('2d');
        pizzaChart1 =new Chart(pizzaCtx, {
            type: 'pie',
            data: {
                labels: data.sentimentos.labels,
                datasets: [{
                    data: data.sentimentos.data,
                    backgroundColor: ['#a61d16', '#807c7c', '#3b7d3c']
                }]
            },
            options: {
                animation: {
                    duration: primeiraAtualizacao ? 1000 : 0
                },
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
        if (pizzaChart2) pizzaChart2.destroy();
        const pizzaCtx2 = document.getElementById('pizzaChart2').getContext('2d');
        pizzaChart2 = new Chart(pizzaCtx2, {
            type: 'pie',
            data: {
                labels: data.sentimentos_modelo.labels.map(label => label.toLowerCase()),
                datasets: [{
                    data: data.sentimentos_modelo.data,
                    backgroundColor: ['#a61d16', '#807c7c', '#3b7d3c']
                }]
            },
            options: {
                animation: {
                    duration: primeiraAtualizacao ? 1000 : 0
                },
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
        if (barChart) barChart.destroy();
        const barCtx = document.getElementById('barChart').getContext('2d');
        barChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: data.analisesPorDia.labels,
                datasets: [
                    {
                        label: 'positivo',
                        data: data.analisesPorDia.positivo,
                        backgroundColor: '#3b7d3c'
                    },
                    {
                        label: 'neutro',
                        data: data.analisesPorDia.neutro,
                        backgroundColor: '#807c7c'
                    },
                    {
                        label: 'negativo',
                        data: data.analisesPorDia.negativo,
                        backgroundColor: '#a61d16'
                    }
                ]
            },
            options: {
                responsive: true,
                animation: {
                    duration: primeiraAtualizacao ? 1000 : 0
                },
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

        primeiraAtualizacao = false;


    } catch (error) {
        console.error('Erro ao carregar dashboard:', error);
    }
}

function exportarParaExcel() {
    fetch("https://api.analisefeedback.com.br/exportar_excel", {
        method: "GET",
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erro ao exportar dados');
        }
        return response.blob(); // Recebe o arquivo
    })
    .then(blob => {
        // Cria um link temporário para baixar o arquivo
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "dados_sentimentos.xlsx";  // Nome do arquivo para download
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error("Erro ao tentar exportar o Excel:", error);
    });
}

window.onload = carregarDashboard;