function validateForm() {
    let dance = document.forms["questionnaire"]["dance"].value;
    let cheer = document.forms["questionnaire"]["cheer"].value;
    let acoustic = document.forms["questionnaire"]["acoustic"].value;
    let lyrics = document.forms["questionnaire"]["lyrics"].value;
    if (dance < 0 || dance > 10 || cheer < 0 || cheer > 10 || acoustic < 0 || acoustic > 10) {
        alert("Oops! Numerical values can only be in range 0 to 10!");
        return false;
    }
    if (lyrics == "") {
        alert("Oops! The main lyrics theme can't be empty!");
        return false;
    }
    if (lyrics.search(" ") == true) {
        alert("Oops! The main lyrics theme can be only one word!");
        return false;
    }
}