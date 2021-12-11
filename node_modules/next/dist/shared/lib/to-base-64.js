"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.toBase64 = toBase64;
function toBase64(str) {
    if (typeof window === 'undefined') {
        return Buffer.from(str).toString('base64');
    } else {
        return window.btoa(str);
    }
}

//# sourceMappingURL=to-base-64.js.map