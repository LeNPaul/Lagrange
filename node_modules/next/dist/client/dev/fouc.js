"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.displayContent = displayContent;
function displayContent(callback) {
    (window.requestAnimationFrame || setTimeout)(function() {
        for(var x = document.querySelectorAll('[data-next-hide-fouc]'), i = x.length; i--;){
            x[i].parentNode.removeChild(x[i]);
        }
        if (callback) {
            callback();
        }
    });
}

//# sourceMappingURL=fouc.js.map