"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.createPagesMapping = createPagesMapping;
exports.createEntrypoints = createEntrypoints;
exports.finalizeEntrypoint = finalizeEntrypoint;
var _chalk = _interopRequireDefault(require("chalk"));
var _path = require("path");
var _querystring = require("querystring");
var _constants = require("../lib/constants");
var _config = require("../server/config");
var _normalizePagePath = require("../server/normalize-page-path");
var _log = require("./output/log");
var _utils = require("./utils");
var _middlewarePlugin = require("./webpack/plugins/middleware-plugin");
var _constants1 = require("../shared/lib/constants");
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
function createPagesMapping(pagePaths, extensions, { isDev , hasServerComponents , hasConcurrentFeatures  }) {
    const previousPages = {
    };
    const pages = pagePaths.reduce((result, pagePath)=>{
        let page = pagePath.replace(new RegExp(`\\.+(${extensions.join('|')})$`), '');
        if (hasServerComponents && /\.client$/.test(page)) {
            // Assume that if there's a Client Component, that there is
            // a matching Server Component that will map to the page.
            return result;
        }
        page = page.replace(/\\/g, '/').replace(/\/index$/, '');
        const pageKey = page === '' ? '/' : page;
        if (pageKey in result) {
            (0, _log).warn(`Duplicate page detected. ${_chalk.default.cyan((0, _path).join('pages', previousPages[pageKey]))} and ${_chalk.default.cyan((0, _path).join('pages', pagePath))} both resolve to ${_chalk.default.cyan(pageKey)}.`);
        } else {
            previousPages[pageKey] = pagePath;
        }
        result[pageKey] = (0, _path).join(_constants.PAGES_DIR_ALIAS, pagePath).replace(/\\/g, '/');
        return result;
    }, {
    });
    // we alias these in development and allow webpack to
    // allow falling back to the correct source file so
    // that HMR can work properly when a file is added/removed
    const documentPage = `_document${hasConcurrentFeatures ? '-web' : ''}`;
    if (isDev) {
        pages['/_app'] = `${_constants.PAGES_DIR_ALIAS}/_app`;
        pages['/_error'] = `${_constants.PAGES_DIR_ALIAS}/_error`;
        pages['/_document'] = `${_constants.PAGES_DIR_ALIAS}/_document`;
    } else {
        pages['/_app'] = pages['/_app'] || 'next/dist/pages/_app';
        pages['/_error'] = pages['/_error'] || 'next/dist/pages/_error';
        pages['/_document'] = pages['/_document'] || `next/dist/pages/${documentPage}`;
    }
    return pages;
}
function createEntrypoints(pages, target, buildId, previewMode, config, loadedEnvFiles) {
    const client = {
    };
    const server = {
    };
    const serverWeb = {
    };
    const hasRuntimeConfig = Object.keys(config.publicRuntimeConfig).length > 0 || Object.keys(config.serverRuntimeConfig).length > 0;
    const defaultServerlessOptions = {
        absoluteAppPath: pages['/_app'],
        absoluteDocumentPath: pages['/_document'],
        absoluteErrorPath: pages['/_error'],
        absolute404Path: pages['/404'] || '',
        distDir: _constants.DOT_NEXT_ALIAS,
        buildId,
        assetPrefix: config.assetPrefix,
        generateEtags: config.generateEtags ? 'true' : '',
        poweredByHeader: config.poweredByHeader ? 'true' : '',
        canonicalBase: config.amp.canonicalBase || '',
        basePath: config.basePath,
        runtimeConfig: hasRuntimeConfig ? JSON.stringify({
            publicRuntimeConfig: config.publicRuntimeConfig,
            serverRuntimeConfig: config.serverRuntimeConfig
        }) : '',
        previewProps: JSON.stringify(previewMode),
        // base64 encode to make sure contents don't break webpack URL loading
        loadedEnvFiles: Buffer.from(JSON.stringify(loadedEnvFiles)).toString('base64'),
        i18n: config.i18n ? JSON.stringify(config.i18n) : ''
    };
    Object.keys(pages).forEach((page)=>{
        const absolutePagePath = pages[page];
        const bundleFile = (0, _normalizePagePath).normalizePagePath(page);
        const isApiRoute = page.match(_constants.API_ROUTE);
        const clientBundlePath = _path.posix.join('pages', bundleFile);
        const serverBundlePath = _path.posix.join('pages', bundleFile);
        const isLikeServerless = (0, _config).isTargetLikeServerless(target);
        const isReserved = (0, _utils).isReservedPage(page);
        const isCustomError = (0, _utils).isCustomErrorPage(page);
        const isFlight = (0, _utils).isFlightPage(config, absolutePagePath);
        const webServerRuntime = !!config.experimental.concurrentFeatures;
        if (page.match(_constants.MIDDLEWARE_ROUTE)) {
            const loaderOpts = {
                absolutePagePath: pages[page],
                page
            };
            client[clientBundlePath] = `next-middleware-loader?${(0, _querystring).stringify(loaderOpts)}!`;
            return;
        }
        if (webServerRuntime && !isReserved && !isCustomError && !isApiRoute) {
            _middlewarePlugin.ssrEntries.set(clientBundlePath, {
                requireFlightManifest: isFlight
            });
            serverWeb[serverBundlePath] = finalizeEntrypoint({
                name: '[name].js',
                value: `next-middleware-ssr-loader?${(0, _querystring).stringify({
                    page,
                    absolute500Path: pages['/500'] || '',
                    absolutePagePath,
                    isServerComponent: isFlight,
                    ...defaultServerlessOptions
                })}!`,
                isServer: false,
                isServerWeb: true
            });
        }
        if (isApiRoute && isLikeServerless) {
            const serverlessLoaderOptions = {
                page,
                absolutePagePath,
                ...defaultServerlessOptions
            };
            server[serverBundlePath] = `next-serverless-loader?${(0, _querystring).stringify(serverlessLoaderOptions)}!`;
        } else if (isApiRoute || target === 'server') {
            if (!webServerRuntime || isReserved || isCustomError) {
                server[serverBundlePath] = [
                    absolutePagePath
                ];
            }
        } else if (isLikeServerless && page !== '/_app' && page !== '/_document' && !webServerRuntime) {
            const serverlessLoaderOptions = {
                page,
                absolutePagePath,
                ...defaultServerlessOptions
            };
            server[serverBundlePath] = `next-serverless-loader?${(0, _querystring).stringify(serverlessLoaderOptions)}!`;
        }
        if (page === '/_document') {
            return;
        }
        if (!isApiRoute) {
            const pageLoaderOpts = {
                page,
                absolutePagePath
            };
            const pageLoader = `next-client-pages-loader?${(0, _querystring).stringify(pageLoaderOpts)}!`;
            // Make sure next/router is a dependency of _app or else chunk splitting
            // might cause the router to not be able to load causing hydration
            // to fail
            client[clientBundlePath] = page === '/_app' ? [
                pageLoader,
                require.resolve('../client/router')
            ] : pageLoader;
        }
    });
    return {
        client,
        server,
        serverWeb
    };
}
function finalizeEntrypoint({ name , value , isServer , isMiddleware , isServerWeb  }) {
    const entry = typeof value !== 'object' || Array.isArray(value) ? {
        import: value
    } : value;
    if (isServer) {
        const isApi = name.startsWith('pages/api/');
        return {
            publicPath: isApi ? '' : undefined,
            runtime: isApi ? 'webpack-api-runtime' : 'webpack-runtime',
            layer: isApi ? 'api' : undefined,
            ...entry
        };
    }
    if (isServerWeb) {
        const ssrMiddlewareEntry = {
            library: {
                name: [
                    '_ENTRIES',
                    `middleware_[name]`
                ],
                type: 'assign'
            },
            runtime: _constants1.MIDDLEWARE_SSR_RUNTIME_WEBPACK,
            asyncChunks: false,
            ...entry
        };
        return ssrMiddlewareEntry;
    }
    if (isMiddleware) {
        const middlewareEntry = {
            filename: 'server/[name].js',
            layer: 'middleware',
            library: {
                name: [
                    '_ENTRIES',
                    `middleware_[name]`
                ],
                type: 'assign'
            },
            ...entry
        };
        return middlewareEntry;
    }
    if (name !== 'polyfills' && name !== 'main' && name !== 'amp' && name !== 'react-refresh') {
        return {
            dependOn: name.startsWith('pages/') && name !== 'pages/_app' ? 'pages/_app' : 'main',
            ...entry
        };
    }
    return entry;
}

//# sourceMappingURL=entries.js.map