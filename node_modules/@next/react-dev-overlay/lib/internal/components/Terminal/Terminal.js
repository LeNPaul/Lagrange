"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
exports.__esModule = true;
exports.Terminal = void 0;
var anser_1 = __importDefault(require("anser"));
var React = __importStar(require("react"));
var Terminal = function Terminal(_a) {
    var content = _a.content;
    var decoded = React.useMemo(function () {
        return anser_1["default"].ansiToJson(content, {
            json: true,
            use_classes: true,
            remove_empty: true
        });
    }, [content]);
    return (React.createElement("div", { "data-nextjs-terminal": true },
        React.createElement("pre", null, decoded.map(function (entry, index) { return (React.createElement("span", { key: "terminal-entry-" + index, style: __assign({ color: entry.fg ? "var(--color-" + entry.fg + ")" : undefined }, (entry.decoration === 'bold'
                ? { fontWeight: 800 }
                : entry.decoration === 'italic'
                    ? { fontStyle: 'italic' }
                    : undefined)) }, entry.content)); }))));
};
exports.Terminal = Terminal;
//# sourceMappingURL=Terminal.js.map