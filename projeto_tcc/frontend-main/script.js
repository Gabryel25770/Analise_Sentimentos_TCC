
function abrirPopup() {
    document.getElementById('popup').style.display = 'flex';
}

  // Função para fechar o popup
function fecharPopup() {
    document.getElementById('popup').style.display = 'none';
}

function abrirAlertNoText() {
    document.getElementById('popup_noText').style.display = 'flex';
}

  // Função para fechar o popup
function fecharAlertNoText() {
    document.getElementById('popup_noText').style.display = 'none';
}

async function teste(){
    let campoTexto = document.getElementById("campotext");
    let texto = campoTexto.value;

    if(texto.length < 1) {
        abrirAlertNoText();
        return;
    }

    if(texto.length < 4) {
        alert("Digite mais para que a análise seja possível!");
        return;
    }

    try {
        const response = await fetch('http://analisesentimentostcc-production.up.railway.app/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: texto })
        });

        const data = await response.json();

        if (data.error) {
            alert("Erro: " + data.error);
            return;
        }

        var sentimento;

        switch(data.sentiment){
            case 'positive':
                sentimento = 'positivo';
                break;
            case 'negative':
                sentimento = 'negativo';
                break;
            case 'neutral':
                sentimento = 'neutro';
                break;
            default:
                break;
        }

        document.getElementById("popup").querySelector("p").innerHTML = `O sentimento detectado foi <strong>${sentimento}</strong>.`;
        abrirPopup();
    } catch (error) {
        alert("Erro ao conectar ao servidor.");
        console.error(error);
    }
}
