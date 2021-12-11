"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
var _getLocaleMetadata = require("../../shared/lib/i18n/get-locale-metadata");
var _cookie = _interopRequireDefault(require("next/dist/compiled/cookie"));
var _router = require("../router");
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
const REGEX_LOCALHOST_HOSTNAME = /(?!^https?:\/\/)(127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}|::1)/;
class NextURL extends URL {
    constructor(input, options = {
    }){
        const url = createWHATWGURL(input);
        super(url);
        this._options = options;
        this._basePath = '';
        this._url = url;
        this.analyzeUrl();
    }
    get absolute() {
        return this._url.hostname !== 'localhost';
    }
    analyzeUrl() {
        const { headers ={
        } , basePath , i18n  } = this._options;
        if (basePath && this._url.pathname.startsWith(basePath)) {
            this._url.pathname = (0, _router).replaceBasePath(this._url.pathname, basePath);
            this._basePath = basePath;
        } else {
            this._basePath = '';
        }
        if (i18n) {
            var ref;
            this._locale = (0, _getLocaleMetadata).getLocaleMetadata({
                cookies: ()=>{
                    const value = headers['cookie'];
                    return value ? _cookie.default.parse(Array.isArray(value) ? value.join(';') : value) : {
                    };
                },
                headers: headers,
                nextConfig: {
                    basePath: basePath,
                    i18n: i18n
                },
                url: {
                    hostname: this._url.hostname || null,
                    pathname: this._url.pathname
                }
            });
            if ((ref = this._locale) === null || ref === void 0 ? void 0 : ref.path.detectedLocale) {
                this._url.pathname = this._locale.path.pathname;
            }
        }
    }
    formatPathname() {
        var ref, ref1;
        const { i18n  } = this._options;
        let pathname = this._url.pathname;
        if (((ref = this._locale) === null || ref === void 0 ? void 0 : ref.locale) && (i18n === null || i18n === void 0 ? void 0 : i18n.defaultLocale) !== ((ref1 = this._locale) === null || ref1 === void 0 ? void 0 : ref1.locale)) {
            var ref5;
            pathname = `/${(ref5 = this._locale) === null || ref5 === void 0 ? void 0 : ref5.locale}${pathname}`;
        }
        if (this._basePath) {
            pathname = `${this._basePath}${pathname}`;
        }
        return pathname;
    }
    get locale() {
        if (!this._locale) {
            throw new TypeError(`The URL is not configured with i18n`);
        }
        return this._locale.locale;
    }
    set locale(locale) {
        if (!this._locale) {
            throw new TypeError(`The URL is not configured with i18n`);
        }
        this._locale.locale = locale;
    }
    get defaultLocale() {
        var ref;
        return (ref = this._locale) === null || ref === void 0 ? void 0 : ref.defaultLocale;
    }
    get domainLocale() {
        var ref;
        return (ref = this._locale) === null || ref === void 0 ? void 0 : ref.domain;
    }
    get searchParams() {
        return this._url.searchParams;
    }
    get host() {
        return this.absolute ? this._url.host : '';
    }
    set host(value) {
        this._url.host = value;
    }
    get hostname() {
        return this.absolute ? this._url.hostname : '';
    }
    set hostname(value) {
        this._url.hostname = value || 'localhost';
    }
    get port() {
        return this.absolute ? this._url.port : '';
    }
    set port(value) {
        this._url.port = value;
    }
    get protocol() {
        return this.absolute ? this._url.protocol : '';
    }
    set protocol(value) {
        this._url.protocol = value;
    }
    get href() {
        const pathname = this.formatPathname();
        return this.absolute ? `${this.protocol}//${this.host}${pathname}${this._url.search}` : `${pathname}${this._url.search}`;
    }
    set href(url) {
        this._url = createWHATWGURL(url);
        this.analyzeUrl();
    }
    get origin() {
        return this.absolute ? this._url.origin : '';
    }
    get pathname() {
        return this._url.pathname;
    }
    set pathname(value) {
        this._url.pathname = value;
    }
    get hash() {
        return this._url.hash;
    }
    set hash(value) {
        this._url.hash = value;
    }
    get search() {
        return this._url.search;
    }
    set search(value) {
        this._url.search = value;
    }
    get password() {
        return this._url.password;
    }
    set password(value) {
        this._url.password = value;
    }
    get username() {
        return this._url.username;
    }
    set username(value) {
        this._url.username = value;
    }
    get basePath() {
        return this._basePath;
    }
    set basePath(value) {
        this._basePath = value.startsWith('/') ? value : `/${value}`;
    }
    toString() {
        return this.href;
    }
    toJSON() {
        return this.href;
    }
}
exports.NextURL = NextURL;
function createWHATWGURL(url) {
    url = url.replace(REGEX_LOCALHOST_HOSTNAME, 'localhost');
    return isRelativeURL(url) ? new URL(url.replace(/^\/+/, '/'), new URL('https://localhost')) : new URL(url);
}
function isRelativeURL(url) {
    return url.startsWith('/');
}

//# sourceMappingURL=next-url.js.map