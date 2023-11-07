let currentQuestionIndex = 1;
let queryParams = "";

function startQuiz() {
    document.querySelector('.quiz-container').classList.add('hidden');
    document.getElementById('question-container').classList.remove('hidden');
}

function nextQuestion(answer) {
    queryParams += answer + "&";

    if (document.getElementById("question" + (currentQuestionIndex + 1))) {
        document
        .getElementById("question" + currentQuestionIndex)
        .classList.add("hidden");
        currentQuestionIndex++;
        document
        .getElementById("question" + currentQuestionIndex)
        .classList.remove("hidden");
    } else {
    // window.location.href = 'result.html?' + queryParams.slice(0, -1); // Removing the trailing "&"
    window.location.href = "/result?" + queryParams.slice(0, -1);

      // Make an HTTP POST request to your Python server
    fetch("/send-data-to-server", {
        method: "POST",
        headers: {
        "Content-Type": "application/json",
        },
        body: JSON.stringify(queryParams),
    })
        .then((response) => response.json())
        .then((result) => {
        // Handle the response from the server (if needed)
        console.log(result);
        })
        .catch((error) => {
        console.error("Error:", error);
        });
    }
}

