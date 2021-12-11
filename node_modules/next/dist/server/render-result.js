"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = void 0;
class RenderResult {
    constructor(response){
        this._result = response;
    }
    toUnchunkedString() {
        if (typeof this._result !== 'string') {
            throw new Error('invariant: dynamic responses cannot be unchunked. This is a bug in Next.js');
        }
        return this._result;
    }
    pipe(res) {
        if (typeof this._result === 'string') {
            throw new Error('invariant: static responses cannot be piped. This is a bug in Next.js');
        }
        const response = this._result;
        return new Promise((resolve, reject)=>{
            response(res, (err)=>err ? reject(err) : resolve()
            );
        });
    }
    isDynamic() {
        return typeof this._result !== 'string';
    }
    static fromStatic(value) {
        return new RenderResult(value);
    }
}
RenderResult.empty = RenderResult.fromStatic('');
exports.default = RenderResult;

//# sourceMappingURL=render-result.js.map