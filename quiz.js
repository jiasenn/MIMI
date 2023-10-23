let currentQuestionIndex = 1;
let queryParams = "";

function startQuiz() {
    document.querySelector('.quiz-container').classList.add('hidden');
    document.getElementById('question-container').classList.remove('hidden');
}

function nextQuestion(answer) {
    queryParams += answer + "&";
    
    if (document.getElementById('question' + (currentQuestionIndex + 1))) {
        document.getElementById('question' + currentQuestionIndex).classList.add('hidden');
        currentQuestionIndex++;
        document.getElementById('question' + currentQuestionIndex).classList.remove('hidden');
    } else {
        window.location.href = 'result.html?' + queryParams.slice(0, -1); // Removing the trailing "&"
    }
}
