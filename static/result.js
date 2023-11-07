function displayResult() {
    // let params = new URLSearchParams(window.location.search);
    // let samsuiCount = 0;
    
    // for(let param of params.values()) {
    //     if(param === 'a') samsuiCount++;
    // }

    // let resultText = '';
    
    // document.getElementById('resultText').innerText = resultText;
    document.getElementById('popup').classList.remove('hidden');
}

function closePopup() {
    document.getElementById('popup').classList.add('hidden');
}
