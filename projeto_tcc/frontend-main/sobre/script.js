
var sentimentoResposta;

function limparTexto(){
    document.getElementById("campotext").value = "";
}

function abrirPopup() {
    document.getElementById('popup').style.display = 'flex';
}

  // Função para fechar o popup
function fecharPopup(event) {
    document.getElementById('popup').style.display = 'none';
    salvarFeedback(event);
}

function abrirAlertNoText() {
    const alerta = document.querySelector('.alert');
    alerta.style.display = 'flex';
    alerta.classList.remove('hide');
    setTimeout(() => {
        alerta.classList.add('hide');

        setTimeout(() => {
          alerta.style.display = 'none';
          alerta.classList.remove('hide');   // Limpa para próxima vez
        }, 500);
      }, 7000);
}

  // Função para fechar o popup
function fecharAlertNoText() {
    const alerta = document.querySelector('.alert');
    alerta.classList.add('hide');
  
    // Espera o tempo da transição para depois dar display: none
    setTimeout(() => {
      alerta.style.display = 'none';
      alerta.classList.remove('hide'); // opcional, caso queira mostrar de novo depois
    }, 500); 
}

function abrirAlertNotEnough() {
    const alerta = document.querySelector('.alert_notEnough');
    alerta.style.display = 'flex';
    alerta.classList.remove('hide');
    setTimeout(() => {
        alerta.classList.add('hide');

        setTimeout(() => {
          alerta.style.display = 'none';
          alerta.classList.remove('hide');   // Limpa para próxima vez
        }, 500);
      }, 7000);
}

  // Função para fechar o popup
function fecharAlertNotEnough() {
    const alerta = document.querySelector('.alert_notEnough');
    alerta.classList.add('hide');
  
    // Espera o tempo da transição para depois dar display: none
    setTimeout(() => {
      alerta.style.display = 'none';
      alerta.classList.remove('hide'); // opcional, caso queira mostrar de novo depois
    }, 500); 
}

function abrirAlertSavedText() {
    const alerta = document.querySelector('.alert_savedText');
    alerta.style.display = 'flex';
    alerta.classList.remove('hide');
    setTimeout(() => {
        alerta.classList.add('hide');

        setTimeout(() => {
          alerta.style.display = 'none';
          alerta.classList.remove('hide');   // Limpa para próxima vez
        }, 500);
      }, 7000);
}

  // Função para fechar o popup
function fecharAlertSavedText() {
    const alerta = document.querySelector('.alert_savedText');
    alerta.classList.add('hide');
  
    // Espera o tempo da transição para depois dar display: none
    setTimeout(() => {
      alerta.style.display = 'none';
      alerta.classList.remove('hide'); // opcional, caso queira mostrar de novo depois
    }, 500); 
}

async function salvarFeedback(event) {
    event.preventDefault();
    const texto = document.getElementById("campotext").value;
    sentimentoResposta = sentimentoResposta.toLowerCase();

    if (!texto || !sentimentoResposta) {
        alert("Texto ou sentimento ausente.");
        return;
    }

    limparTexto();

    try {
        console.log("entrou no try")
        const response = await fetch("http://127.0.0.1:5000/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: texto, sentiment: sentimentoResposta })
        });

        const data = await response.json();

        if (data.error) {
            alert("Erro ao salvar: " + data.error);
        } else {
            console.log("Feedback salvo com sucesso.");
        }

    } catch (error) {
        console.error("Erro ao salvar feedback:", error);
    }
}

function clickYes(){
    document.getElementById("pergunta-popup").innerHTML = `Ótimo! Muito obrigado por sua participação.`;

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;
}

function clickNo(){
    document.getElementById("pergunta-popup").innerHTML = `Certo, então qual destas opções melhor classifica seu feedback?`;

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `flex`;
    document.getElementById("div-success").style.display = `none`;
}

function clickPos(){
    if(sentimentoResposta == 'POSITIVO')
        document.getElementById("pergunta-popup").innerHTML = `Entendi, talvez tenhamos acertado. Muito obrigado pela participação!`;
    else{
        document.getElementById("pergunta-popup").innerHTML = `Anotado, muito obrigado pela ajuda e pela participação!`;
    }

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;

    sentimentoResposta = 'positivo'
}

function clickNeg(){
    if(sentimentoResposta == 'NEGATIVO')
        document.getElementById("pergunta-popup").innerHTML = `Entendi, talvez tenhamos acertado. Muito obrigado pela participação!`;
    else{
        document.getElementById("pergunta-popup").innerHTML = `Anotado, muito obrigado pela ajuda e pela participação!`;
    }

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;

    sentimentoResposta = 'negativo'
}

function clickNtl(){
    if(sentimentoResposta == 'NEUTRO')
        document.getElementById("pergunta-popup").innerHTML = `Entendi, talvez tenhamos acertado. Muito obrigado pela participação!`;
    else{
        document.getElementById("pergunta-popup").innerHTML = `Anotado, muito obrigado pela ajuda e pela participação!`;
    }

    document.getElementById("div-sim-nao").style.display = `none`;
    document.getElementById("div-options").style.display = `none`;
    document.getElementById("div-success").style.display = `flex`;

    sentimentoResposta = 'neutro'
}

async function analise(){
    let campoTexto = document.getElementById("campotext");
    let texto = campoTexto.value;

    if(texto.length < 1) {
        abrirAlertNoText();
        return;
    }

    if(texto.length < 4) {
        abrirAlertNotEnough();
        return;
    }

    // Mostra carregamento
    document.getElementById("loading").style.display = 'flex';

    try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: texto })
        });

        const data = await response.json();
        console.log("Data", data);
        //console.log("Modelo_t5", data.modelo_t5)

        if (data.error) {
            alert("Erro: " + data.error);
            return;
        }

        var sentimento;
        var image;
        var color;

        switch(data.modelo_t5){
            case 'positive':
                sentimento = 'POSITIVO';
                image = '../assets/positive-vote.png';
                color = '#3b7d3c'
                break;
            case 'negative':
                sentimento = 'NEGATIVO';
                image = '../assets/negative-vote.png';
                color = '#a61d16'
                break;
            case 'neutral':
                sentimento = 'NEUTRO';
                image = '../assets/line.png';
                color = '#807c7c'
                break;
            default:
                break;
        }

        sentimentoResposta = '';
        sentimentoResposta = sentimento
        

        document.getElementById("div-sim-nao").style.display = `flex`;
        document.getElementById("div-options").style.display = `none`;
        document.getElementById("div-success").style.display = `none`;

        document.getElementById("pergunta-popup").innerHTML = `A análise do seu texto está correta?`;
        document.getElementById("responseText").innerHTML = `<strong>${sentimento}</strong>`;
        document.getElementById("responseText").style.color = `${color}`;
        document.getElementById("header-popup").style["background-color"] = `${color}`;
        document.getElementById("imageFeedback").src = `${image}`
        abrirPopup();

    } catch (error) {
        alert("Erro ao conectar ao servidor.");
        console.error(error);
    } finally {
        // Esconde carregamento
        document.getElementById("loading").style.display = 'none';
    }
}

