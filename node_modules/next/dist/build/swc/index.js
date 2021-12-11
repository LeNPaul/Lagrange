"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.isWasm = isWasm;
exports.transform = transform;
exports.transformSync = transformSync;
exports.minify = minify;
exports.minifySync = minifySync;
exports.bundle = bundle;
var _os = require("os");
var _triples = require("@napi-rs/triples");
var Log = _interopRequireWildcard(require("../output/log"));
function _interopRequireWildcard(obj) {
    if (obj && obj.__esModule) {
        return obj;
    } else {
        var newObj = {
        };
        if (obj != null) {
            for(var key in obj){
                if (Object.prototype.hasOwnProperty.call(obj, key)) {
                    var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {
                    };
                    if (desc.get || desc.set) {
                        Object.defineProperty(newObj, key, desc);
                    } else {
                        newObj[key] = obj[key];
                    }
                }
            }
        }
        newObj.default = obj;
        return newObj;
    }
}
const ArchName = (0, _os).arch();
const PlatformName = (0, _os).platform();
const triples = _triples.platformArchTriples[PlatformName][ArchName] || [];
async function loadBindings() {
    return await loadWasm() || loadNative();
}
async function loadWasm() {
    // Try to load wasm bindings
    for (let specifier of [
        '@next/swc-wasm-web',
        '@next/swc-wasm-nodejs'
    ]){
        try {
            let bindings = await import(specifier);
            if (specifier === '@next/swc-wasm-web') {
                bindings = await bindings.default();
            }
            return {
                isWasm: true,
                transform (src, options) {
                    return Promise.resolve(bindings.transformSync(src.toString(), options));
                },
                minify (src, options) {
                    return Promise.resolve(bindings.minifySync(src.toString(), options));
                }
            };
        } catch (e) {
        }
    }
}
function loadNative() {
    let bindings;
    let loadError;
    for (const triple of triples){
        try {
            bindings = require(`@next/swc/native/next-swc.${triple.platformArchABI}.node`);
            Log.info('Using locally built binary of @next/swc');
            break;
        } catch (e) {
            if ((e === null || e === void 0 ? void 0 : e.code) !== 'MODULE_NOT_FOUND') {
                loadError = e;
            }
        }
    }
    if (!bindings) {
        for (const triple of triples){
            try {
                bindings = require(`@next/swc-${triple.platformArchABI}`);
                break;
            } catch (e) {
                if ((e === null || e === void 0 ? void 0 : e.code) !== 'MODULE_NOT_FOUND') {
                    loadError = e;
                }
            }
        }
    }
    if (bindings) {
        return {
            isWasm: false,
            transform (src, options) {
                var ref;
                const isModule = typeof src !== undefined && typeof src !== 'string' && !Buffer.isBuffer(src);
                options = options || {
                };
                if (options === null || options === void 0 ? void 0 : (ref = options.jsc) === null || ref === void 0 ? void 0 : ref.parser) {
                    var _syntax;
                    options.jsc.parser.syntax = (_syntax = options.jsc.parser.syntax) !== null && _syntax !== void 0 ? _syntax : 'ecmascript';
                }
                return bindings.transform(isModule ? JSON.stringify(src) : src, isModule, toBuffer(options));
            },
            transformSync (src, options) {
                var ref;
                if (typeof src === undefined) {
                    throw new Error("transformSync doesn't implement reading the file from filesystem");
                } else if (Buffer.isBuffer(src)) {
                    throw new Error("transformSync doesn't implement taking the source code as Buffer");
                }
                const isModule = typeof src !== 'string';
                options = options || {
                };
                if (options === null || options === void 0 ? void 0 : (ref = options.jsc) === null || ref === void 0 ? void 0 : ref.parser) {
                    var _syntax;
                    options.jsc.parser.syntax = (_syntax = options.jsc.parser.syntax) !== null && _syntax !== void 0 ? _syntax : 'ecmascript';
                }
                return bindings.transformSync(isModule ? JSON.stringify(src) : src, isModule, toBuffer(options));
            },
            minify (src, options) {
                return bindings.minify(toBuffer(src), toBuffer(options !== null && options !== void 0 ? options : {
                }));
            },
            minifySync (src, options) {
                return bindings.minifySync(toBuffer(src), toBuffer(options !== null && options !== void 0 ? options : {
                }));
            },
            bundle (options) {
                return bindings.bundle(toBuffer(options));
            }
        };
    }
    if (loadError) {
        console.error(loadError);
    }
    Log.error(`Failed to load SWC binary, see more info here: https://nextjs.org/docs/messages/failed-loading-swc`);
    process.exit(1);
}
function toBuffer(t) {
    return Buffer.from(JSON.stringify(t));
}
async function isWasm() {
    let bindings = await loadBindings();
    return bindings.isWasm;
}
async function transform(src, options) {
    let bindings = await loadBindings();
    return bindings.transform(src, options);
}
function transformSync(src, options) {
    let bindings = loadNative();
    return bindings.transformSync(src, options);
}
async function minify(src, options) {
    let bindings = await loadBindings();
    return bindings.minify(src, options);
}
function minifySync(src, options) {
    let bindings = loadNative();
    return bindings.minifySync(src, options);
}
async function bundle(options) {
    let bindings = loadNative();
    return bindings.bundle(toBuffer(options));
}

//# sourceMappingURL=index.js.map