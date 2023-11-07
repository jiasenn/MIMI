function displayResult() {
    let params = new URLSearchParams(window.location.search);
    let samsuiCount = 0;
    
    for(let param of params.values()) {
        if(param === 'a') samsuiCount++;
    }

    let resultText = samsuiCount > 1 ? 'You are more of a Samsui Woman! The caregiver who symbolises diligence and sacrifice.' : 'You are more of a Coolie! The everyman who embodies hardship and perseverance.';
    
    document.getElementById('resultText').innerText = resultText;
    document.getElementById('popup').classList.remove('hidden');
}

function closePopup() {
    document.getElementById('popup').classList.add('hidden');
}
