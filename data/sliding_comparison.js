function slider(event, synced=false) {
    var activeContainer = event.currentTarget;
    var activeRight = activeContainer.querySelector(".right");
    var offset = activeRight.getBoundingClientRect().left;

    if (synced) {
        sliders = document.getElementsByClassName("slider");
    } else {
        sliders = [event.currentTarget];
    }
    for (var i = 0; i < sliders.length; i++) {
        var right = sliders[i].querySelector(".right");
        var position = ((event.pageX - offset) / right.offsetWidth) * 100;
        right.style.clipPath = "inset(0px 0px 0px " + (position) + "%)";
    }
}