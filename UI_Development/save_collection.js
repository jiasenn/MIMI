document.getElementById('saveToCollection').addEventListener('click', function() {
    var imageUrl = document.getElementById('resultImage').src; // Ensure the image has this ID.
    var collection = JSON.parse(localStorage.getItem('collection')) || [];
    collection.push(imageUrl);
    localStorage.setItem('collection', JSON.stringify(collection));
    localStorage.setItem('savedToCollection', 'true'); // Set the flag.
    alert('Image added to collection!');
    window.location.href = 'profile.html'; // Redirect to the profile page.
});