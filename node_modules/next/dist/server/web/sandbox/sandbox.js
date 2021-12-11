"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.run = run;
var _context = require("./context");
async function run(params) {
    const { runInContext , context  } = (0, _context).getModuleContext({
        module: params.name,
        onWarning: params.onWarning,
        useCache: params.useCache !== false
    });
    for (const paramPath of params.paths){
        runInContext(paramPath);
    }
    return context._ENTRIES[`middleware_${params.name}`].default({
        request: params.request
    });
}

//# sourceMappingURL=sandbox.js.map