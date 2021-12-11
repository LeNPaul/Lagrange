"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = isError;
function isError(err) {
    return typeof err === 'object' && err !== null && 'name' in err && 'message' in err;
}

//# sourceMappingURL=is-error.js.map