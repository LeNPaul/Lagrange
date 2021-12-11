"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.atob = atob;
exports.btoa = btoa;
Object.defineProperty(exports, "CryptoKey", {
    enumerable: true,
    get: function() {
        return _webcrypto.CryptoKey;
    }
});
var _webcrypto = require("next/dist/compiled/@peculiar/webcrypto");
var _webStreamsPolyfill = require("next/dist/compiled/web-streams-polyfill");
var _uuid = require("next/dist/compiled/uuid");
var _crypto = _interopRequireDefault(require("crypto"));
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
function atob(b64Encoded) {
    return Buffer.from(b64Encoded, 'base64').toString('binary');
}
function btoa(str) {
    return Buffer.from(str, 'binary').toString('base64');
}
class Crypto extends _webcrypto.Crypto {
    constructor(...args){
        super(...args);
        // @ts-ignore Remove once types are updated and we deprecate node 12
        this.randomUUID = _crypto.default.randomUUID || _uuid.v4;
    }
}
exports.Crypto = Crypto;
class ReadableStream {
    constructor(opts = {
    }){
        let closed = false;
        let pullPromise;
        let transformController;
        const { readable , writable  } = new _webStreamsPolyfill.TransformStream({
            start: (controller)=>{
                transformController = controller;
            }
        }, undefined, {
            highWaterMark: 1
        });
        const writer = writable.getWriter();
        const encoder = new TextEncoder();
        const controller = {
            get desiredSize () {
                return transformController.desiredSize;
            },
            close: ()=>{
                if (!closed) {
                    closed = true;
                    writer.close();
                }
            },
            enqueue: (chunk)=>{
                writer.write(typeof chunk === 'string' ? encoder.encode(chunk) : chunk);
                pull();
            },
            error: (reason)=>{
                transformController.error(reason);
            }
        };
        const pull = ()=>{
            if (opts.pull) {
                if (!pullPromise) {
                    pullPromise = Promise.resolve().then(()=>{
                        pullPromise = 0;
                        opts.pull(controller);
                    });
                }
            }
        };
        if (opts.start) {
            opts.start(controller);
        }
        if (opts.cancel) {
            readable.cancel = (reason)=>{
                opts.cancel(reason);
                return readable.cancel(reason);
            };
        }
        pull();
        return readable;
    }
}
exports.ReadableStream = ReadableStream;

//# sourceMappingURL=polyfills.js.map