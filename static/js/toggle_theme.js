function toggleTheme() {
    let themeLink = document.getElementById('theme-style');
    if (themeLink.getAttribute('href').includes('style.css')) {
        themeLink.setAttribute('href', '/static/css/dark.css');
    } else {
        themeLink.setAttribute('href', '/static/css/style.css');
    }
}
