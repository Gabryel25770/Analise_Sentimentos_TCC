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

function teste(){
    let campoTexto = document.getElementById("campotext");
    let texto = campoTexto.value;

    if(texto.length < 1) {
        abrirAlertNoText()
        return
    }

    if(texto.length < 4) {
        alert("Digite mais para que a analise seja possível!")
        return
    }

    abrirPopup()

    console.log("Resultado: ", texto)
}


