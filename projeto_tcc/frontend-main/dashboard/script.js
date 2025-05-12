function navigateTo(page) {
    window.location.href = page;
}

async function carregarDashboard() {
    try {
        // Rota que vai fornecer os dados:
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
                    backgroundColor: ['#4CAF50', '#FF6384', '#36A2EB', '#FFCE56', '#AA66CC']
                }]
            }
        });

        // --- Atualiza tabela ---
        const tabelaBody = document.querySelector('#registrosTable tbody');
        tabelaBody.innerHTML = "";
        data.registros.forEach(registro => {
            const row = `<tr>
                <td>${registro.id}</td>
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
                datasets: [{
                    label: 'Análises',
                    data: data.analisesPorDia.data,
                    backgroundColor: '#36A2EB'
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        precision: 0
                    }
                }
            }
        });

    } catch (error) {
        console.error('Erro ao carregar dashboard:', error);
    }
}

window.onload = carregarDashboard;