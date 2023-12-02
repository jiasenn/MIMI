document.addEventListener('DOMContentLoaded', function() {
    var container = document.getElementById('collectionContainer');
    if (container) {
        var collection = JSON.parse(localStorage.getItem('collection')) || [];
        collection.forEach(function(imageUrl) {
            var img = document.createElement('img');
            img.src = imageUrl;
            img.classList.add('collectedImage'); // Add class for styling if needed
            container.appendChild(img);
        });
    }
});

document.querySelector(".clear").addEventListener("click", function () {
  localStorage.clear();
});