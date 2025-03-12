

function teste(){
    let campoTexto = document.getElementById("campotext");
    let texto = campoTexto.value;

    if(texto.length < 1) {
        alert("Digite algum texto para ser analisado!")
        return
    }

    if(texto.length < 4) {
        alert("Digite mais para que a analise seja possível!")
        return
    }

    console.log("Resultado: ", texto)
}


