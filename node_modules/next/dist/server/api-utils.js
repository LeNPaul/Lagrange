"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.apiResolver = apiResolver;
exports.parseBody = parseBody;
exports.getCookieParser = getCookieParser;
exports.sendStatusCode = sendStatusCode;
exports.redirect = redirect;
exports.sendData = sendData;
exports.sendJson = sendJson;
exports.tryGetPreviewData = tryGetPreviewData;
exports.sendError = sendError;
exports.setLazyProp = setLazyProp;
exports.SYMBOL_PREVIEW_DATA = void 0;
var _contentType = require("next/dist/compiled/content-type");
var _rawBody = _interopRequireDefault(require("raw-body"));
var _stream = require("stream");
var _utils = require("../shared/lib/utils");
var _cryptoUtils = require("./crypto-utils");
var _sendPayload = require("./send-payload");
var _etag = _interopRequireDefault(require("etag"));
var _isError = _interopRequireDefault(require("../lib/is-error"));
var _interopDefault = require("../lib/interop-default");
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
async function apiResolver(req, res, query, resolverModule, apiContext, propagateError, dev, page) {
    const apiReq = req;
    const apiRes = res;
    try {
        var ref, ref1;
        if (!resolverModule) {
            res.statusCode = 404;
            res.end('Not Found');
            return;
        }
        const config = resolverModule.config || {
        };
        const bodyParser = ((ref = config.api) === null || ref === void 0 ? void 0 : ref.bodyParser) !== false;
        const externalResolver = ((ref1 = config.api) === null || ref1 === void 0 ? void 0 : ref1.externalResolver) || false;
        // Parsing of cookies
        setLazyProp({
            req: apiReq
        }, 'cookies', getCookieParser(req.headers));
        // Parsing query string
        apiReq.query = query;
        // Parsing preview data
        setLazyProp({
            req: apiReq
        }, 'previewData', ()=>tryGetPreviewData(req, res, apiContext)
        );
        // Checking if preview mode is enabled
        setLazyProp({
            req: apiReq
        }, 'preview', ()=>apiReq.previewData !== false ? true : undefined
        );
        // Parsing of body
        if (bodyParser && !apiReq.body) {
            apiReq.body = await parseBody(apiReq, config.api && config.api.bodyParser && config.api.bodyParser.sizeLimit ? config.api.bodyParser.sizeLimit : '1mb');
        }
        let contentLength = 0;
        const writeData = apiRes.write;
        const endResponse = apiRes.end;
        apiRes.write = (...args)=>{
            contentLength += Buffer.byteLength(args[0] || '');
            return writeData.apply(apiRes, args);
        };
        apiRes.end = (...args)=>{
            if (args.length && typeof args[0] !== 'function') {
                contentLength += Buffer.byteLength(args[0] || '');
            }
            if (contentLength >= 4 * 1024 * 1024) {
                console.warn(`API response for ${req.url} exceeds 4MB. This will cause the request to fail in a future version. https://nextjs.org/docs/messages/api-routes-body-size-limit`);
            }
            endResponse.apply(apiRes, args);
        };
        apiRes.status = (statusCode)=>sendStatusCode(apiRes, statusCode)
        ;
        apiRes.send = (data)=>sendData(apiReq, apiRes, data)
        ;
        apiRes.json = (data)=>sendJson(apiRes, data)
        ;
        apiRes.redirect = (statusOrUrl, url)=>redirect(apiRes, statusOrUrl, url)
        ;
        apiRes.setPreviewData = (data, options = {
        })=>setPreviewData(apiRes, data, Object.assign({
            }, apiContext, options))
        ;
        apiRes.clearPreviewData = ()=>clearPreviewData(apiRes)
        ;
        const resolver = (0, _interopDefault).interopDefault(resolverModule);
        let wasPiped = false;
        if (process.env.NODE_ENV !== 'production') {
            // listen for pipe event and don't show resolve warning
            res.once('pipe', ()=>wasPiped = true
            );
        }
        // Call API route method
        await resolver(req, res);
        if (process.env.NODE_ENV !== 'production' && !externalResolver && !(0, _utils).isResSent(res) && !wasPiped) {
            console.warn(`API resolved without sending a response for ${req.url}, this may result in stalled requests.`);
        }
    } catch (err) {
        if (err instanceof ApiError) {
            sendError(apiRes, err.statusCode, err.message);
        } else {
            if (dev) {
                if ((0, _isError).default(err)) {
                    err.page = page;
                }
                throw err;
            }
            console.error(err);
            if (propagateError) {
                throw err;
            }
            sendError(apiRes, 500, 'Internal Server Error');
        }
    }
}
async function parseBody(req, limit) {
    let contentType;
    try {
        contentType = (0, _contentType).parse(req.headers['content-type'] || 'text/plain');
    } catch  {
        contentType = (0, _contentType).parse('text/plain');
    }
    const { type , parameters  } = contentType;
    const encoding = parameters.charset || 'utf-8';
    let buffer;
    try {
        buffer = await (0, _rawBody).default(req, {
            encoding,
            limit
        });
    } catch (e) {
        if ((0, _isError).default(e) && e.type === 'entity.too.large') {
            throw new ApiError(413, `Body exceeded ${limit} limit`);
        } else {
            throw new ApiError(400, 'Invalid body');
        }
    }
    const body = buffer.toString();
    if (type === 'application/json' || type === 'application/ld+json') {
        return parseJson(body);
    } else if (type === 'application/x-www-form-urlencoded') {
        const qs = require('querystring');
        return qs.decode(body);
    } else {
        return body;
    }
}
/**
 * Parse `JSON` and handles invalid `JSON` strings
 * @param str `JSON` string
 */ function parseJson(str) {
    if (str.length === 0) {
        // special-case empty json body, as it's a common client-side mistake
        return {
        };
    }
    try {
        return JSON.parse(str);
    } catch (e) {
        throw new ApiError(400, 'Invalid JSON');
    }
}
function getCookieParser(headers) {
    return function parseCookie() {
        const header = headers.cookie;
        if (!header) {
            return {
            };
        }
        const { parse: parseCookieFn  } = require('next/dist/compiled/cookie');
        return parseCookieFn(Array.isArray(header) ? header.join(';') : header);
    };
}
function sendStatusCode(res, statusCode) {
    res.statusCode = statusCode;
    return res;
}
function redirect(res, statusOrUrl, url) {
    if (typeof statusOrUrl === 'string') {
        url = statusOrUrl;
        statusOrUrl = 307;
    }
    if (typeof statusOrUrl !== 'number' || typeof url !== 'string') {
        throw new Error(`Invalid redirect arguments. Please use a single argument URL, e.g. res.redirect('/destination') or use a status code and URL, e.g. res.redirect(307, '/destination').`);
    }
    res.writeHead(statusOrUrl, {
        Location: url
    });
    res.write(url);
    res.end();
    return res;
}
function sendData(req, res, body) {
    if (body === null || body === undefined) {
        res.end();
        return;
    }
    // strip irrelevant headers/body
    if (res.statusCode === 204 || res.statusCode === 304) {
        res.removeHeader('Content-Type');
        res.removeHeader('Content-Length');
        res.removeHeader('Transfer-Encoding');
        if (process.env.NODE_ENV === 'development' && body) {
            console.warn(`A body was attempted to be set with a 204 statusCode for ${req.url}, this is invalid and the body was ignored.\n` + `See more info here https://nextjs.org/docs/messages/invalid-api-status-body`);
        }
        res.end();
        return;
    }
    const contentType = res.getHeader('Content-Type');
    if (body instanceof _stream.Stream) {
        if (!contentType) {
            res.setHeader('Content-Type', 'application/octet-stream');
        }
        body.pipe(res);
        return;
    }
    const isJSONLike = [
        'object',
        'number',
        'boolean'
    ].includes(typeof body);
    const stringifiedBody = isJSONLike ? JSON.stringify(body) : body;
    const etag = (0, _etag).default(stringifiedBody);
    if ((0, _sendPayload).sendEtagResponse(req, res, etag)) {
        return;
    }
    if (Buffer.isBuffer(body)) {
        if (!contentType) {
            res.setHeader('Content-Type', 'application/octet-stream');
        }
        res.setHeader('Content-Length', body.length);
        res.end(body);
        return;
    }
    if (isJSONLike) {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
    }
    res.setHeader('Content-Length', Buffer.byteLength(stringifiedBody));
    res.end(stringifiedBody);
}
function sendJson(res, jsonBody) {
    // Set header to application/json
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    // Use send to handle request
    res.send(jsonBody);
}
const COOKIE_NAME_PRERENDER_BYPASS = `__prerender_bypass`;
const COOKIE_NAME_PRERENDER_DATA = `__next_preview_data`;
const SYMBOL_PREVIEW_DATA = Symbol(COOKIE_NAME_PRERENDER_DATA);
exports.SYMBOL_PREVIEW_DATA = SYMBOL_PREVIEW_DATA;
const SYMBOL_CLEARED_COOKIES = Symbol(COOKIE_NAME_PRERENDER_BYPASS);
function tryGetPreviewData(req, res, options) {
    // Read cached preview data if present
    if (SYMBOL_PREVIEW_DATA in req) {
        return req[SYMBOL_PREVIEW_DATA];
    }
    const getCookies = getCookieParser(req.headers);
    let cookies;
    try {
        cookies = getCookies();
    } catch  {
        // TODO: warn
        return false;
    }
    const hasBypass = COOKIE_NAME_PRERENDER_BYPASS in cookies;
    const hasData = COOKIE_NAME_PRERENDER_DATA in cookies;
    // Case: neither cookie is set.
    if (!(hasBypass || hasData)) {
        return false;
    }
    // Case: one cookie is set, but not the other.
    if (hasBypass !== hasData) {
        clearPreviewData(res);
        return false;
    }
    // Case: preview session is for an old build.
    if (cookies[COOKIE_NAME_PRERENDER_BYPASS] !== options.previewModeId) {
        clearPreviewData(res);
        return false;
    }
    const tokenPreviewData = cookies[COOKIE_NAME_PRERENDER_DATA];
    const jsonwebtoken = require('next/dist/compiled/jsonwebtoken');
    let encryptedPreviewData;
    try {
        encryptedPreviewData = jsonwebtoken.verify(tokenPreviewData, options.previewModeSigningKey);
    } catch  {
        // TODO: warn
        clearPreviewData(res);
        return false;
    }
    const decryptedPreviewData = (0, _cryptoUtils).decryptWithSecret(Buffer.from(options.previewModeEncryptionKey), encryptedPreviewData.data);
    try {
        // TODO: strict runtime type checking
        const data = JSON.parse(decryptedPreviewData);
        // Cache lookup
        Object.defineProperty(req, SYMBOL_PREVIEW_DATA, {
            value: data,
            enumerable: false
        });
        return data;
    } catch  {
        return false;
    }
}
function isNotValidData(str) {
    return typeof str !== 'string' || str.length < 16;
}
function setPreviewData(res, data, options) {
    if (isNotValidData(options.previewModeId)) {
        throw new Error('invariant: invalid previewModeId');
    }
    if (isNotValidData(options.previewModeEncryptionKey)) {
        throw new Error('invariant: invalid previewModeEncryptionKey');
    }
    if (isNotValidData(options.previewModeSigningKey)) {
        throw new Error('invariant: invalid previewModeSigningKey');
    }
    const jsonwebtoken = require('next/dist/compiled/jsonwebtoken');
    const payload = jsonwebtoken.sign({
        data: (0, _cryptoUtils).encryptWithSecret(Buffer.from(options.previewModeEncryptionKey), JSON.stringify(data))
    }, options.previewModeSigningKey, {
        algorithm: 'HS256',
        ...options.maxAge !== undefined ? {
            expiresIn: options.maxAge
        } : undefined
    });
    // limit preview mode cookie to 2KB since we shouldn't store too much
    // data here and browsers drop cookies over 4KB
    if (payload.length > 2048) {
        throw new Error(`Preview data is limited to 2KB currently, reduce how much data you are storing as preview data to continue`);
    }
    const { serialize  } = require('next/dist/compiled/cookie');
    const previous = res.getHeader('Set-Cookie');
    res.setHeader(`Set-Cookie`, [
        ...typeof previous === 'string' ? [
            previous
        ] : Array.isArray(previous) ? previous : [],
        serialize(COOKIE_NAME_PRERENDER_BYPASS, options.previewModeId, {
            httpOnly: true,
            sameSite: process.env.NODE_ENV !== 'development' ? 'none' : 'lax',
            secure: process.env.NODE_ENV !== 'development',
            path: '/',
            ...options.maxAge !== undefined ? {
                maxAge: options.maxAge
            } : undefined
        }),
        serialize(COOKIE_NAME_PRERENDER_DATA, payload, {
            httpOnly: true,
            sameSite: process.env.NODE_ENV !== 'development' ? 'none' : 'lax',
            secure: process.env.NODE_ENV !== 'development',
            path: '/',
            ...options.maxAge !== undefined ? {
                maxAge: options.maxAge
            } : undefined
        }), 
    ]);
    return res;
}
function clearPreviewData(res) {
    if (SYMBOL_CLEARED_COOKIES in res) {
        return res;
    }
    const { serialize  } = require('next/dist/compiled/cookie');
    const previous = res.getHeader('Set-Cookie');
    res.setHeader(`Set-Cookie`, [
        ...typeof previous === 'string' ? [
            previous
        ] : Array.isArray(previous) ? previous : [],
        serialize(COOKIE_NAME_PRERENDER_BYPASS, '', {
            // To delete a cookie, set `expires` to a date in the past:
            // https://tools.ietf.org/html/rfc6265#section-4.1.1
            // `Max-Age: 0` is not valid, thus ignored, and the cookie is persisted.
            expires: new Date(0),
            httpOnly: true,
            sameSite: process.env.NODE_ENV !== 'development' ? 'none' : 'lax',
            secure: process.env.NODE_ENV !== 'development',
            path: '/'
        }),
        serialize(COOKIE_NAME_PRERENDER_DATA, '', {
            // To delete a cookie, set `expires` to a date in the past:
            // https://tools.ietf.org/html/rfc6265#section-4.1.1
            // `Max-Age: 0` is not valid, thus ignored, and the cookie is persisted.
            expires: new Date(0),
            httpOnly: true,
            sameSite: process.env.NODE_ENV !== 'development' ? 'none' : 'lax',
            secure: process.env.NODE_ENV !== 'development',
            path: '/'
        }), 
    ]);
    Object.defineProperty(res, SYMBOL_CLEARED_COOKIES, {
        value: true,
        enumerable: false
    });
    return res;
}
class ApiError extends Error {
    constructor(statusCode, message){
        super(message);
        this.statusCode = statusCode;
    }
}
exports.ApiError = ApiError;
function sendError(res, statusCode, message) {
    res.statusCode = statusCode;
    res.statusMessage = message;
    res.end(message);
}
function setLazyProp({ req  }, prop, getter) {
    const opts = {
        configurable: true,
        enumerable: true
    };
    const optsReset = {
        ...opts,
        writable: true
    };
    Object.defineProperty(req, prop, {
        ...opts,
        get: ()=>{
            const value = getter();
            // we set the property on the object to avoid recalculating it
            Object.defineProperty(req, prop, {
                ...optsReset,
                value
            });
            return value;
        },
        set: (value)=>{
            Object.defineProperty(req, prop, {
                ...optsReset,
                value
            });
        }
    });
}

//# sourceMappingURL=api-utils.js.map