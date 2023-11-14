document.addEventListener('DOMContentLoaded', function () {
    // Get the current URL path and file name
    var path = window.location.pathname;
    var currentPage = path.substring(path.lastIndexOf('/') + 1); // Extract just the file name

    // Function to set the active tab based on the current page
    function setActiveTab(currentPage) {
        // Your tabs' ID or any unique part of the path to match
        var tabs = {
            'index.html': 'home-tab',
            'explore.html': 'explore-tab',
            'collection.html': 'collection-tab',
            'profile.html': 'profile-tab'
        };

        // Remove 'active' class from all tabs first
        Object.values(tabs).forEach(function(tabId) {
            var tab = document.getElementById(tabId);
            tab.classList.remove('active');
            var defaultIcon = tab.querySelector('img:not(.activeIcon)');
            var activeIcon = tab.querySelector('img.activeIcon');
            if (defaultIcon) defaultIcon.style.display = 'block';
            if (activeIcon) activeIcon.style.display = 'none';
        });

        // Get the ID of the current tab
        var currentTabId = tabs[currentPage];

        // Get the current tab and add the 'active' class
        var currentTab = document.getElementById(currentTabId);
        if (currentTab) {
            currentTab.classList.add('active');
            // Show the active icon and hide the default
            var defaultIcon = currentTab.querySelector('img:not(.activeIcon)');
            var activeIcon = currentTab.querySelector('img.activeIcon');
            if (defaultIcon) defaultIcon.style.display = 'none';
            if (activeIcon) activeIcon.style.display = 'block';
        }
    }

    // Set the active tab based on the current path
    setActiveTab(currentPage);
});
