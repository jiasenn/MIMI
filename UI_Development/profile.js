document.addEventListener('DOMContentLoaded', function() {
    if (localStorage.getItem('savedToCollection') === 'true') {
        document.getElementById('archetypeSection').style.display = 'block';
        localStorage.removeItem('savedToCollection'); // Clear the flag.
    }
});