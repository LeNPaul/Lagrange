"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.renderToHTML = renderToHTML;
exports.useMaybeDeferContent = useMaybeDeferContent;
var _react = _interopRequireDefault(require("react"));
var _server = _interopRequireDefault(require("react-dom/server"));
var _reactServerDomWebpack = require("next/dist/compiled/react-server-dom-webpack");
var _writerBrowserServer = require("next/dist/compiled/react-server-dom-webpack/writer.browser.server");
var _styledJsx = require("styled-jsx");
var _constants = require("../lib/constants");
var _isSerializableProps = require("../lib/is-serializable-props");
var _amp = require("../shared/lib/amp");
var _ampContext = require("../shared/lib/amp-context");
var _constants1 = require("../shared/lib/constants");
var _head = require("../shared/lib/head");
var _headManagerContext = require("../shared/lib/head-manager-context");
var _loadable = _interopRequireDefault(require("../shared/lib/loadable"));
var _loadableContext = require("../shared/lib/loadable-context");
var _routerContext = require("../shared/lib/router-context");
var _isDynamic = require("../shared/lib/router/utils/is-dynamic");
var _utils = require("../shared/lib/utils");
var _denormalizePagePath = require("./denormalize-page-path");
var _normalizePagePath = require("./normalize-page-path");
var _requestMeta = require("./request-meta");
var _loadCustomRoutes = require("../lib/load-custom-routes");
var _renderResult = _interopRequireDefault(require("./render-result"));
var _isError = _interopRequireDefault(require("../lib/is-error"));
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
let Writable;
let Buffer;
let optimizeAmp;
let getFontDefinitionFromManifest;
let tryGetPreviewData;
let warn;
let postProcess;
const DOCTYPE = '<!DOCTYPE html>';
if (!process.browser) {
    Writable = require('stream').Writable;
    Buffer = require('buffer').Buffer;
    optimizeAmp = require('./optimize-amp').default;
    getFontDefinitionFromManifest = require('./font-utils').getFontDefinitionFromManifest;
    tryGetPreviewData = require('./api-utils').tryGetPreviewData;
    warn = require('../build/output/log').warn;
    postProcess = require('../shared/lib/post-process').default;
} else {
    warn = console.warn.bind(console);
}
function noRouter() {
    const message = 'No router instance found. you should only use "next/router" inside the client side of your app. https://nextjs.org/docs/messages/no-router-instance';
    throw new Error(message);
}
class ServerRouter {
    constructor(pathname, query, as, { isFallback  }, isReady, basePath, locale, locales, defaultLocale, domainLocales, isPreview, isLocaleDomain){
        this.route = pathname.replace(/\/$/, '') || '/';
        this.pathname = pathname;
        this.query = query;
        this.asPath = as;
        this.isFallback = isFallback;
        this.basePath = basePath;
        this.locale = locale;
        this.locales = locales;
        this.defaultLocale = defaultLocale;
        this.isReady = isReady;
        this.domainLocales = domainLocales;
        this.isPreview = !!isPreview;
        this.isLocaleDomain = !!isLocaleDomain;
    }
    push() {
        noRouter();
    }
    replace() {
        noRouter();
    }
    reload() {
        noRouter();
    }
    back() {
        noRouter();
    }
    prefetch() {
        noRouter();
    }
    beforePopState() {
        noRouter();
    }
}
function enhanceComponents(options, App, Component) {
    // For backwards compatibility
    if (typeof options === 'function') {
        return {
            App,
            Component: options(Component)
        };
    }
    return {
        App: options.enhanceApp ? options.enhanceApp(App) : App,
        Component: options.enhanceComponent ? options.enhanceComponent(Component) : Component
    };
}
const invalidKeysMsg = (methodName, invalidKeys)=>{
    return `Additional keys were returned from \`${methodName}\`. Properties intended for your component must be nested under the \`props\` key, e.g.:` + `\n\n\treturn { props: { title: 'My Title', content: '...' } }` + `\n\nKeys that need to be moved: ${invalidKeys.join(', ')}.` + `\nRead more: https://nextjs.org/docs/messages/invalid-getstaticprops-value`;
};
function checkRedirectValues(redirect, req, method) {
    const { destination , permanent , statusCode , basePath  } = redirect;
    let errors = [];
    const hasStatusCode = typeof statusCode !== 'undefined';
    const hasPermanent = typeof permanent !== 'undefined';
    if (hasPermanent && hasStatusCode) {
        errors.push(`\`permanent\` and \`statusCode\` can not both be provided`);
    } else if (hasPermanent && typeof permanent !== 'boolean') {
        errors.push(`\`permanent\` must be \`true\` or \`false\``);
    } else if (hasStatusCode && !_loadCustomRoutes.allowedStatusCodes.has(statusCode)) {
        errors.push(`\`statusCode\` must undefined or one of ${[
            ..._loadCustomRoutes.allowedStatusCodes
        ].join(', ')}`);
    }
    const destinationType = typeof destination;
    if (destinationType !== 'string') {
        errors.push(`\`destination\` should be string but received ${destinationType}`);
    }
    const basePathType = typeof basePath;
    if (basePathType !== 'undefined' && basePathType !== 'boolean') {
        errors.push(`\`basePath\` should be undefined or a false, received ${basePathType}`);
    }
    if (errors.length > 0) {
        throw new Error(`Invalid redirect object returned from ${method} for ${req.url}\n` + errors.join(' and ') + '\n' + `See more info here: https://nextjs.org/docs/messages/invalid-redirect-gssp`);
    }
}
// Create the wrapper component for a Flight stream.
function createServerComponentRenderer(OriginalComponent, serverComponentManifest) {
    let responseCache;
    const ServerComponentWrapper = (props)=>{
        let response = responseCache;
        if (!response) {
            responseCache = response = (0, _reactServerDomWebpack).createFromReadableStream((0, _writerBrowserServer).renderToReadableStream(/*#__PURE__*/ _react.default.createElement(OriginalComponent, Object.assign({
            }, props)), serverComponentManifest));
        }
        return response.readRoot();
    };
    const Component = (props)=>{
        return(/*#__PURE__*/ _react.default.createElement(_react.default.Suspense, {
            fallback: null
        }, /*#__PURE__*/ _react.default.createElement(ServerComponentWrapper, Object.assign({
        }, props))));
    };
    // Although it's not allowed to attach some static methods to Component,
    // we still re-assign all the component APIs to keep the behavior unchanged.
    for (const methodName of [
        'getInitialProps',
        'getStaticProps',
        'getServerSideProps',
        'getStaticPaths', 
    ]){
        const method = OriginalComponent[methodName];
        if (method) {
            Component[methodName] = method;
        }
    }
    return Component;
}
async function renderToHTML(req, res, pathname, query, renderOpts) {
    // In dev we invalidate the cache by appending a timestamp to the resource URL.
    // This is a workaround to fix https://github.com/vercel/next.js/issues/5860
    // TODO: remove this workaround when https://bugs.webkit.org/show_bug.cgi?id=187726 is fixed.
    renderOpts.devOnlyCacheBusterQueryString = renderOpts.dev ? renderOpts.devOnlyCacheBusterQueryString || `?ts=${Date.now()}` : '';
    // don't modify original query object
    query = Object.assign({
    }, query);
    const { err , dev =false , ampPath ='' , App , Document , pageConfig ={
    } , buildManifest , fontManifest , reactLoadableManifest , ErrorDebug , getStaticProps , getStaticPaths , getServerSideProps , serverComponentManifest , renderServerComponentData , isDataReq , params , previewProps , basePath , devOnlyCacheBusterQueryString , supportsDynamicHTML , concurrentFeatures ,  } = renderOpts;
    const isServerComponent = !!serverComponentManifest;
    const OriginalComponent = renderOpts.Component;
    const Component = isServerComponent ? createServerComponentRenderer(OriginalComponent, serverComponentManifest) : renderOpts.Component;
    const getFontDefinition = (url)=>{
        if (fontManifest) {
            return getFontDefinitionFromManifest(url, fontManifest);
        }
        return '';
    };
    const callMiddleware = async (method, args, props = false)=>{
        let results = props ? {
        } : [];
        if (Document[`${method}Middleware`]) {
            let middlewareFunc = await Document[`${method}Middleware`];
            middlewareFunc = middlewareFunc.default || middlewareFunc;
            const curResults = await middlewareFunc(...args);
            if (props) {
                for (const result of curResults){
                    results = {
                        ...results,
                        ...result
                    };
                }
            } else {
                results = curResults;
            }
        }
        return results;
    };
    const headTags = (...args)=>callMiddleware('headTags', args)
    ;
    const isFallback = !!query.__nextFallback;
    delete query.__nextFallback;
    delete query.__nextLocale;
    delete query.__nextDefaultLocale;
    const isSSG = !!getStaticProps;
    const isBuildTimeSSG = isSSG && renderOpts.nextExport;
    const defaultAppGetInitialProps = App.getInitialProps === App.origGetInitialProps;
    const hasPageGetInitialProps = !!Component.getInitialProps;
    const pageIsDynamic = (0, _isDynamic).isDynamicRoute(pathname);
    const isAutoExport = !hasPageGetInitialProps && defaultAppGetInitialProps && !isSSG && !getServerSideProps && !concurrentFeatures;
    for (const methodName of [
        'getStaticProps',
        'getServerSideProps',
        'getStaticPaths', 
    ]){
        if (Component[methodName]) {
            throw new Error(`page ${pathname} ${methodName} ${_constants.GSSP_COMPONENT_MEMBER_ERROR}`);
        }
    }
    if (hasPageGetInitialProps && isSSG) {
        throw new Error(_constants.SSG_GET_INITIAL_PROPS_CONFLICT + ` ${pathname}`);
    }
    if (hasPageGetInitialProps && getServerSideProps) {
        throw new Error(_constants.SERVER_PROPS_GET_INIT_PROPS_CONFLICT + ` ${pathname}`);
    }
    if (getServerSideProps && isSSG) {
        throw new Error(_constants.SERVER_PROPS_SSG_CONFLICT + ` ${pathname}`);
    }
    if (getStaticPaths && !pageIsDynamic) {
        throw new Error(`getStaticPaths is only allowed for dynamic SSG pages and was found on '${pathname}'.` + `\nRead more: https://nextjs.org/docs/messages/non-dynamic-getstaticpaths-usage`);
    }
    if (!!getStaticPaths && !isSSG) {
        throw new Error(`getStaticPaths was added without a getStaticProps in ${pathname}. Without getStaticProps, getStaticPaths does nothing`);
    }
    if (isSSG && pageIsDynamic && !getStaticPaths) {
        throw new Error(`getStaticPaths is required for dynamic SSG pages and is missing for '${pathname}'.` + `\nRead more: https://nextjs.org/docs/messages/invalid-getstaticpaths-value`);
    }
    let asPath = renderOpts.resolvedAsPath || req.url;
    if (dev) {
        const { isValidElementType  } = require('react-is');
        if (!isValidElementType(Component)) {
            throw new Error(`The default export is not a React Component in page: "${pathname}"`);
        }
        if (!isValidElementType(App)) {
            throw new Error(`The default export is not a React Component in page: "/_app"`);
        }
        if (!isValidElementType(Document)) {
            throw new Error(`The default export is not a React Component in page: "/_document"`);
        }
        if (isAutoExport || isFallback) {
            // remove query values except ones that will be set during export
            query = {
                ...query.amp ? {
                    amp: query.amp
                } : {
                }
            };
            asPath = `${pathname}${// ensure trailing slash is present for non-dynamic auto-export pages
            req.url.endsWith('/') && pathname !== '/' && !pageIsDynamic ? '/' : ''}`;
            req.url = pathname;
        }
        if (pathname === '/404' && (hasPageGetInitialProps || getServerSideProps)) {
            throw new Error(`\`pages/404\` ${_constants.STATIC_STATUS_PAGE_GET_INITIAL_PROPS_ERROR}`);
        }
        if (_constants1.STATIC_STATUS_PAGES.includes(pathname) && (hasPageGetInitialProps || getServerSideProps)) {
            throw new Error(`\`pages${pathname}\` ${_constants.STATIC_STATUS_PAGE_GET_INITIAL_PROPS_ERROR}`);
        }
    }
    await _loadable.default.preloadAll() // Make sure all dynamic imports are loaded
    ;
    let isPreview;
    let previewData;
    if ((isSSG || getServerSideProps) && !isFallback && !process.browser) {
        // Reads of this are cached on the `req` object, so this should resolve
        // instantly. There's no need to pass this data down from a previous
        // invoke, where we'd have to consider server & serverless.
        previewData = tryGetPreviewData(req, res, previewProps);
        isPreview = previewData !== false;
    }
    // url will always be set
    const routerIsReady = !!(getServerSideProps || hasPageGetInitialProps || !defaultAppGetInitialProps && !isSSG);
    const router = new ServerRouter(pathname, query, asPath, {
        isFallback: isFallback
    }, routerIsReady, basePath, renderOpts.locale, renderOpts.locales, renderOpts.defaultLocale, renderOpts.domainLocales, isPreview, (0, _requestMeta).getRequestMeta(req, '__nextIsLocaleDomain'));
    const jsxStyleRegistry = (0, _styledJsx).createStyleRegistry();
    const ctx = {
        err,
        req: isAutoExport ? undefined : req,
        res: isAutoExport ? undefined : res,
        pathname,
        query,
        asPath,
        locale: renderOpts.locale,
        locales: renderOpts.locales,
        defaultLocale: renderOpts.defaultLocale,
        AppTree: (props)=>{
            return(/*#__PURE__*/ _react.default.createElement(AppContainerWithIsomorphicFiberStructure, null, /*#__PURE__*/ _react.default.createElement(App, Object.assign({
            }, props, {
                Component: Component,
                router: router
            }))));
        },
        defaultGetInitialProps: async (docCtx)=>{
            const enhanceApp = (AppComp)=>{
                return (props)=>/*#__PURE__*/ _react.default.createElement(AppComp, Object.assign({
                    }, props))
                ;
            };
            const { html , head  } = await docCtx.renderPage({
                enhanceApp
            });
            const styles = jsxStyleRegistry.styles();
            return {
                html,
                head,
                styles
            };
        }
    };
    let props;
    const ampState = {
        ampFirst: pageConfig.amp === true,
        hasQuery: Boolean(query.amp),
        hybrid: pageConfig.amp === 'hybrid'
    };
    // Disable AMP under the web environment
    const inAmpMode = !process.browser && (0, _amp).isInAmpMode(ampState);
    const reactLoadableModules = [];
    let head = (0, _head).defaultHead(inAmpMode);
    let scriptLoader = {
    };
    const nextExport = !isSSG && (renderOpts.nextExport || dev && (isAutoExport || isFallback));
    const AppContainer = ({ children  })=>/*#__PURE__*/ _react.default.createElement(_routerContext.RouterContext.Provider, {
            value: router
        }, /*#__PURE__*/ _react.default.createElement(_ampContext.AmpStateContext.Provider, {
            value: ampState
        }, /*#__PURE__*/ _react.default.createElement(_headManagerContext.HeadManagerContext.Provider, {
            value: {
                updateHead: (state)=>{
                    head = state;
                },
                updateScripts: (scripts)=>{
                    scriptLoader = scripts;
                },
                scripts: {
                },
                mountedInstances: new Set()
            }
        }, /*#__PURE__*/ _react.default.createElement(_loadableContext.LoadableContext.Provider, {
            value: (moduleName)=>reactLoadableModules.push(moduleName)
        }, /*#__PURE__*/ _react.default.createElement(_styledJsx.StyleRegistry, {
            registry: jsxStyleRegistry
        }, children)))))
    ;
    // The `useId` API uses the path indexes to generate an ID for each node.
    // To guarantee the match of hydration, we need to ensure that the structure
    // of wrapper nodes is isomorphic in server and client.
    // TODO: With `enhanceApp` and `enhanceComponents` options, this approach may
    // not be useful.
    // https://github.com/facebook/react/pull/22644
    const Noop = ()=>null
    ;
    const AppContainerWithIsomorphicFiberStructure = ({ children  })=>{
        return(/*#__PURE__*/ _react.default.createElement(_react.default.Fragment, null, /*#__PURE__*/ _react.default.createElement(Noop, null), /*#__PURE__*/ _react.default.createElement(AppContainer, null, /*#__PURE__*/ _react.default.createElement(_react.default.Fragment, null, dev ? /*#__PURE__*/ _react.default.createElement(_react.default.Fragment, null, children, /*#__PURE__*/ _react.default.createElement(Noop, null)) : children, /*#__PURE__*/ _react.default.createElement(Noop, null)))));
    };
    props = await (0, _utils).loadGetInitialProps(App, {
        AppTree: ctx.AppTree,
        Component,
        router,
        ctx
    });
    if ((isSSG || getServerSideProps) && isPreview) {
        props.__N_PREVIEW = true;
    }
    if (isSSG) {
        props[_constants1.STATIC_PROPS_ID] = true;
    }
    if (isSSG && !isFallback) {
        let data;
        try {
            data = await getStaticProps({
                ...pageIsDynamic ? {
                    params: query
                } : undefined,
                ...isPreview ? {
                    preview: true,
                    previewData: previewData
                } : undefined,
                locales: renderOpts.locales,
                locale: renderOpts.locale,
                defaultLocale: renderOpts.defaultLocale
            });
        } catch (staticPropsError) {
            // remove not found error code to prevent triggering legacy
            // 404 rendering
            if (staticPropsError && staticPropsError.code === 'ENOENT') {
                delete staticPropsError.code;
            }
            throw staticPropsError;
        }
        if (data == null) {
            throw new Error(_constants.GSP_NO_RETURNED_VALUE);
        }
        const invalidKeys = Object.keys(data).filter((key)=>key !== 'revalidate' && key !== 'props' && key !== 'redirect' && key !== 'notFound'
        );
        if (invalidKeys.includes('unstable_revalidate')) {
            throw new Error(_constants.UNSTABLE_REVALIDATE_RENAME_ERROR);
        }
        if (invalidKeys.length) {
            throw new Error(invalidKeysMsg('getStaticProps', invalidKeys));
        }
        if (process.env.NODE_ENV !== 'production') {
            if (typeof data.notFound !== 'undefined' && typeof data.redirect !== 'undefined') {
                throw new Error(`\`redirect\` and \`notFound\` can not both be returned from ${isSSG ? 'getStaticProps' : 'getServerSideProps'} at the same time. Page: ${pathname}\nSee more info here: https://nextjs.org/docs/messages/gssp-mixed-not-found-redirect`);
            }
        }
        if ('notFound' in data && data.notFound) {
            if (pathname === '/404') {
                throw new Error(`The /404 page can not return notFound in "getStaticProps", please remove it to continue!`);
            }
            renderOpts.isNotFound = true;
        }
        if ('redirect' in data && data.redirect && typeof data.redirect === 'object') {
            checkRedirectValues(data.redirect, req, 'getStaticProps');
            if (isBuildTimeSSG) {
                throw new Error(`\`redirect\` can not be returned from getStaticProps during prerendering (${req.url})\n` + `See more info here: https://nextjs.org/docs/messages/gsp-redirect-during-prerender`);
            }
            data.props = {
                __N_REDIRECT: data.redirect.destination,
                __N_REDIRECT_STATUS: (0, _loadCustomRoutes).getRedirectStatus(data.redirect)
            };
            if (typeof data.redirect.basePath !== 'undefined') {
                data.props.__N_REDIRECT_BASE_PATH = data.redirect.basePath;
            }
            renderOpts.isRedirect = true;
        }
        if ((dev || isBuildTimeSSG) && !renderOpts.isNotFound && !(0, _isSerializableProps).isSerializableProps(pathname, 'getStaticProps', data.props)) {
            // this fn should throw an error instead of ever returning `false`
            throw new Error('invariant: getStaticProps did not return valid props. Please report this.');
        }
        if ('revalidate' in data) {
            if (typeof data.revalidate === 'number') {
                if (!Number.isInteger(data.revalidate)) {
                    throw new Error(`A page's revalidate option must be seconds expressed as a natural number for ${req.url}. Mixed numbers, such as '${data.revalidate}', cannot be used.` + `\nTry changing the value to '${Math.ceil(data.revalidate)}' or using \`Math.ceil()\` if you're computing the value.`);
                } else if (data.revalidate <= 0) {
                    throw new Error(`A page's revalidate option can not be less than or equal to zero for ${req.url}. A revalidate option of zero means to revalidate after _every_ request, and implies stale data cannot be tolerated.` + `\n\nTo never revalidate, you can set revalidate to \`false\` (only ran once at build-time).` + `\nTo revalidate as soon as possible, you can set the value to \`1\`.`);
                } else if (data.revalidate > 31536000) {
                    // if it's greater than a year for some reason error
                    console.warn(`Warning: A page's revalidate option was set to more than a year for ${req.url}. This may have been done in error.` + `\nTo only run getStaticProps at build-time and not revalidate at runtime, you can set \`revalidate\` to \`false\`!`);
                }
            } else if (data.revalidate === true) {
                // When enabled, revalidate after 1 second. This value is optimal for
                // the most up-to-date page possible, but without a 1-to-1
                // request-refresh ratio.
                data.revalidate = 1;
            } else if (data.revalidate === false || typeof data.revalidate === 'undefined') {
                // By default, we never revalidate.
                data.revalidate = false;
            } else {
                throw new Error(`A page's revalidate option must be seconds expressed as a natural number. Mixed numbers and strings cannot be used. Received '${JSON.stringify(data.revalidate)}' for ${req.url}`);
            }
        } else {
            data.revalidate = false;
        }
        props.pageProps = Object.assign({
        }, props.pageProps, 'props' in data ? data.props : undefined);
        renderOpts.revalidate = 'revalidate' in data ? data.revalidate : undefined;
        renderOpts.pageData = props;
        // this must come after revalidate is added to renderOpts
        if (renderOpts.isNotFound) {
            return null;
        }
    }
    if (getServerSideProps) {
        props[_constants1.SERVER_PROPS_ID] = true;
    }
    if (getServerSideProps && !isFallback) {
        let data;
        let canAccessRes = true;
        let resOrProxy = res;
        let deferredContent = false;
        if (process.env.NODE_ENV !== 'production') {
            resOrProxy = new Proxy(res, {
                get: function(obj, prop, receiver) {
                    if (!canAccessRes) {
                        const message = `You should not access 'res' after getServerSideProps resolves.` + `\nRead more: https://nextjs.org/docs/messages/gssp-no-mutating-res`;
                        if (deferredContent) {
                            throw new Error(message);
                        } else {
                            warn(message);
                        }
                    }
                    return Reflect.get(obj, prop, receiver);
                }
            });
        }
        try {
            data = await getServerSideProps({
                req: req,
                res: resOrProxy,
                query,
                resolvedUrl: renderOpts.resolvedUrl,
                ...pageIsDynamic ? {
                    params: params
                } : undefined,
                ...previewData !== false ? {
                    preview: true,
                    previewData: previewData
                } : undefined,
                locales: renderOpts.locales,
                locale: renderOpts.locale,
                defaultLocale: renderOpts.defaultLocale
            });
            canAccessRes = false;
        } catch (serverSidePropsError) {
            // remove not found error code to prevent triggering legacy
            // 404 rendering
            if ((0, _isError).default(serverSidePropsError) && serverSidePropsError.code === 'ENOENT') {
                delete serverSidePropsError.code;
            }
            throw serverSidePropsError;
        }
        if (data == null) {
            throw new Error(_constants.GSSP_NO_RETURNED_VALUE);
        }
        if (data.props instanceof Promise) {
            deferredContent = true;
        }
        const invalidKeys = Object.keys(data).filter((key)=>key !== 'props' && key !== 'redirect' && key !== 'notFound'
        );
        if (data.unstable_notFound) {
            throw new Error(`unstable_notFound has been renamed to notFound, please update the field to continue. Page: ${pathname}`);
        }
        if (data.unstable_redirect) {
            throw new Error(`unstable_redirect has been renamed to redirect, please update the field to continue. Page: ${pathname}`);
        }
        if (invalidKeys.length) {
            throw new Error(invalidKeysMsg('getServerSideProps', invalidKeys));
        }
        if ('notFound' in data && data.notFound) {
            if (pathname === '/404') {
                throw new Error(`The /404 page can not return notFound in "getStaticProps", please remove it to continue!`);
            }
            renderOpts.isNotFound = true;
            return null;
        }
        if ('redirect' in data && typeof data.redirect === 'object') {
            checkRedirectValues(data.redirect, req, 'getServerSideProps');
            data.props = {
                __N_REDIRECT: data.redirect.destination,
                __N_REDIRECT_STATUS: (0, _loadCustomRoutes).getRedirectStatus(data.redirect)
            };
            if (typeof data.redirect.basePath !== 'undefined') {
                data.props.__N_REDIRECT_BASE_PATH = data.redirect.basePath;
            }
            renderOpts.isRedirect = true;
        }
        if (deferredContent) {
            data.props = await data.props;
        }
        if ((dev || isBuildTimeSSG) && !(0, _isSerializableProps).isSerializableProps(pathname, 'getServerSideProps', data.props)) {
            // this fn should throw an error instead of ever returning `false`
            throw new Error('invariant: getServerSideProps did not return valid props. Please report this.');
        }
        props.pageProps = Object.assign({
        }, props.pageProps, data.props);
        renderOpts.pageData = props;
    }
    if (!isSSG && !getServerSideProps && process.env.NODE_ENV !== 'production' && Object.keys((props === null || props === void 0 ? void 0 : props.pageProps) || {
    }).includes('url')) {
        console.warn(`The prop \`url\` is a reserved prop in Next.js for legacy reasons and will be overridden on page ${pathname}\n` + `See more info here: https://nextjs.org/docs/messages/reserved-page-prop`);
    }
    // Avoid rendering page un-necessarily for getServerSideProps data request
    // and getServerSideProps/getStaticProps redirects
    if (isDataReq && !isSSG || renderOpts.isRedirect) {
        return _renderResult.default.fromStatic(JSON.stringify(props));
    }
    // We don't call getStaticProps or getServerSideProps while generating
    // the fallback so make sure to set pageProps to an empty object
    if (isFallback) {
        props.pageProps = {
        };
    }
    // Pass router to the Server Component as a temporary workaround.
    if (isServerComponent) {
        props.pageProps = Object.assign({
        }, props.pageProps, {
            router
        });
    }
    // the response might be finished on the getInitialProps call
    if ((0, _utils).isResSent(res) && !isSSG) return null;
    if (renderServerComponentData) {
        const stream = (0, _writerBrowserServer).renderToReadableStream(/*#__PURE__*/ _react.default.createElement(OriginalComponent, Object.assign({
        }, props.pageProps)), serverComponentManifest);
        const reader = stream.getReader();
        const piper = (innerRes, next)=>{
            bufferedReadFromReadableStream(reader, (val)=>innerRes.write(val)
            ).then(()=>next()
            , (innerErr)=>next(innerErr)
            );
        };
        return new _renderResult.default(chainPipers([
            piper
        ]));
    }
    // we preload the buildManifest for auto-export dynamic pages
    // to speed up hydrating query values
    let filteredBuildManifest = buildManifest;
    if (isAutoExport && pageIsDynamic) {
        const page = (0, _denormalizePagePath).denormalizePagePath((0, _normalizePagePath).normalizePagePath(pathname));
        // This code would be much cleaner using `immer` and directly pushing into
        // the result from `getPageFiles`, we could maybe consider that in the
        // future.
        if (page in filteredBuildManifest.pages) {
            filteredBuildManifest = {
                ...filteredBuildManifest,
                pages: {
                    ...filteredBuildManifest.pages,
                    [page]: [
                        ...filteredBuildManifest.pages[page],
                        ...filteredBuildManifest.lowPriorityFiles.filter((f)=>f.includes('_buildManifest')
                        ), 
                    ]
                },
                lowPriorityFiles: filteredBuildManifest.lowPriorityFiles.filter((f)=>!f.includes('_buildManifest')
                )
            };
        }
    }
    const Body = ({ children  })=>{
        return inAmpMode ? children : /*#__PURE__*/ _react.default.createElement("div", {
            id: "__next"
        }, children);
    };
    const appWrappers = [];
    const getWrappedApp = (app)=>{
        // Prevent wrappers from reading/writing props by rendering inside an
        // opaque component. Wrappers should use context instead.
        const InnerApp = ()=>app
        ;
        return(/*#__PURE__*/ _react.default.createElement(AppContainerWithIsomorphicFiberStructure, null, appWrappers.reduceRight((innerContent, fn)=>{
            return fn(innerContent);
        }, /*#__PURE__*/ _react.default.createElement(InnerApp, null))));
    };
    /**
   * Rules of Static & Dynamic HTML:
   *
   *    1.) We must generate static HTML unless the caller explicitly opts
   *        in to dynamic HTML support.
   *
   *    2.) If dynamic HTML support is requested, we must honor that request
   *        or throw an error. It is the sole responsibility of the caller to
   *        ensure they aren't e.g. requesting dynamic HTML for an AMP page.
   *
   * These rules help ensure that other existing features like request caching,
   * coalescing, and ISR continue working as intended.
   */ const generateStaticHTML = supportsDynamicHTML !== true;
    const renderDocument = async ()=>{
        if (process.browser && Document.getInitialProps) {
            throw new Error('`getInitialProps` in Document component is not supported with `concurrentFeatures` enabled.');
        }
        if (!process.browser && Document.getInitialProps) {
            const renderPage = (options = {
            })=>{
                if (ctx.err && ErrorDebug) {
                    const html = _server.default.renderToString(/*#__PURE__*/ _react.default.createElement(Body, null, /*#__PURE__*/ _react.default.createElement(ErrorDebug, {
                        error: ctx.err
                    })));
                    return {
                        html,
                        head
                    };
                }
                if (dev && (props.router || props.Component)) {
                    throw new Error(`'router' and 'Component' can not be returned in getInitialProps from _app.js https://nextjs.org/docs/messages/cant-override-next-props`);
                }
                const { App: EnhancedApp , Component: EnhancedComponent  } = enhanceComponents(options, App, Component);
                const html = _server.default.renderToString(/*#__PURE__*/ _react.default.createElement(Body, null, getWrappedApp(/*#__PURE__*/ _react.default.createElement(EnhancedApp, Object.assign({
                    Component: EnhancedComponent,
                    router: router
                }, props)))));
                return {
                    html,
                    head
                };
            };
            const documentCtx = {
                ...ctx,
                renderPage
            };
            const docProps = await (0, _utils).loadGetInitialProps(Document, documentCtx);
            // the response might be finished on the getInitialProps call
            if ((0, _utils).isResSent(res) && !isSSG) return null;
            if (!docProps || typeof docProps.html !== 'string') {
                const message = `"${(0, _utils).getDisplayName(Document)}.getInitialProps()" should resolve to an object with a "html" prop set with a valid html string`;
                throw new Error(message);
            }
            return {
                bodyResult: ()=>piperFromArray([
                        docProps.html
                    ])
                ,
                documentElement: (htmlProps)=>/*#__PURE__*/ _react.default.createElement(Document, Object.assign({
                    }, htmlProps, docProps))
                ,
                useMainContent: (fn)=>{
                    if (fn) {
                        throw new Error('The `children` property is not supported by non-functional custom Document components');
                    }
                    // @ts-ignore
                    return(/*#__PURE__*/ _react.default.createElement("next-js-internal-body-render-target", null));
                },
                head: docProps.head,
                headTags: await headTags(documentCtx),
                styles: docProps.styles
            };
        } else {
            let bodyResult;
            if (concurrentFeatures) {
                bodyResult = async ()=>{
                    // this must be called inside bodyResult so appWrappers is
                    // up to date when getWrappedApp is called
                    const content = ctx.err && ErrorDebug ? /*#__PURE__*/ _react.default.createElement(Body, null, /*#__PURE__*/ _react.default.createElement(ErrorDebug, {
                        error: ctx.err
                    })) : /*#__PURE__*/ _react.default.createElement(Body, null, getWrappedApp(/*#__PURE__*/ _react.default.createElement(App, Object.assign({
                    }, props, {
                        Component: Component,
                        router: router
                    }))));
                    return process.browser ? await renderToWebStream(content) : await renderToNodeStream(content, generateStaticHTML);
                };
            } else {
                const content = ctx.err && ErrorDebug ? /*#__PURE__*/ _react.default.createElement(Body, null, /*#__PURE__*/ _react.default.createElement(ErrorDebug, {
                    error: ctx.err
                })) : /*#__PURE__*/ _react.default.createElement(Body, null, getWrappedApp(/*#__PURE__*/ _react.default.createElement(App, Object.assign({
                }, props, {
                    Component: Component,
                    router: router
                }))));
                // for non-concurrent rendering we need to ensure App is rendered
                // before _document so that updateHead is called/collected before
                // rendering _document's head
                const result = piperFromArray([
                    _server.default.renderToString(content)
                ]);
                bodyResult = ()=>result
                ;
            }
            return {
                bodyResult,
                documentElement: ()=>Document()
                ,
                useMainContent: (fn)=>{
                    if (fn) {
                        appWrappers.push(fn);
                    }
                    // @ts-ignore
                    return(/*#__PURE__*/ _react.default.createElement("next-js-internal-body-render-target", null));
                },
                head,
                headTags: [],
                styles: jsxStyleRegistry.styles()
            };
        }
    };
    const documentResult = await renderDocument();
    if (!documentResult) {
        return null;
    }
    const dynamicImportsIds = new Set();
    const dynamicImports = new Set();
    for (const mod of reactLoadableModules){
        const manifestItem = reactLoadableManifest[mod];
        if (manifestItem) {
            dynamicImportsIds.add(manifestItem.id);
            manifestItem.files.forEach((item)=>{
                dynamicImports.add(item);
            });
        }
    }
    const hybridAmp = ampState.hybrid;
    const docComponentsRendered = {
    };
    const { assetPrefix , buildId , customServer , defaultLocale , disableOptimizedLoading , domainLocales , locale , locales , runtimeConfig ,  } = renderOpts;
    const htmlProps = {
        __NEXT_DATA__: {
            props,
            page: pathname,
            query,
            buildId,
            assetPrefix: assetPrefix === '' ? undefined : assetPrefix,
            runtimeConfig,
            nextExport: nextExport === true ? true : undefined,
            autoExport: isAutoExport === true ? true : undefined,
            isFallback,
            dynamicIds: dynamicImportsIds.size === 0 ? undefined : Array.from(dynamicImportsIds),
            err: renderOpts.err ? serializeError(dev, renderOpts.err) : undefined,
            gsp: !!getStaticProps ? true : undefined,
            gssp: !!getServerSideProps ? true : undefined,
            rsc: isServerComponent ? true : undefined,
            customServer,
            gip: hasPageGetInitialProps ? true : undefined,
            appGip: !defaultAppGetInitialProps ? true : undefined,
            locale,
            locales,
            defaultLocale,
            domainLocales,
            isPreview: isPreview === true ? true : undefined
        },
        buildManifest: filteredBuildManifest,
        docComponentsRendered,
        dangerousAsPath: router.asPath,
        canonicalBase: !renderOpts.ampPath && (0, _requestMeta).getRequestMeta(req, '__nextStrippedLocale') ? `${renderOpts.canonicalBase || ''}/${renderOpts.locale}` : renderOpts.canonicalBase,
        ampPath,
        inAmpMode,
        isDevelopment: !!dev,
        hybridAmp,
        dynamicImports: Array.from(dynamicImports),
        assetPrefix,
        // Only enabled in production as development mode has features relying on HMR (style injection for example)
        unstable_runtimeJS: process.env.NODE_ENV === 'production' ? pageConfig.unstable_runtimeJS : undefined,
        unstable_JsPreload: pageConfig.unstable_JsPreload,
        devOnlyCacheBusterQueryString,
        scriptLoader,
        locale,
        disableOptimizedLoading,
        head: documentResult.head,
        headTags: documentResult.headTags,
        styles: documentResult.styles,
        useMainContent: documentResult.useMainContent,
        useMaybeDeferContent,
        crossOrigin: renderOpts.crossOrigin,
        optimizeCss: renderOpts.optimizeCss,
        optimizeFonts: renderOpts.optimizeFonts,
        optimizeImages: renderOpts.optimizeImages,
        concurrentFeatures: renderOpts.concurrentFeatures
    };
    const document = /*#__PURE__*/ _react.default.createElement(_ampContext.AmpStateContext.Provider, {
        value: ampState
    }, /*#__PURE__*/ _react.default.createElement(_utils.HtmlContext.Provider, {
        value: htmlProps
    }, documentResult.documentElement(htmlProps)));
    let documentHTML;
    if (process.browser) {
        // There is no `renderToStaticMarkup` exposed in the web environment, use
        // blocking `renderToReadableStream` to get the similar result.
        let result = '';
        const readable = _server.default.renderToReadableStream(document, {
            onError: (e)=>{
                throw e;
            }
        });
        const reader = readable.getReader();
        const decoder = new TextDecoder();
        while(true){
            const { done , value  } = await reader.read();
            if (done) {
                break;
            }
            result += typeof value === 'string' ? value : decoder.decode(value);
        }
        documentHTML = result;
    } else {
        documentHTML = _server.default.renderToStaticMarkup(document);
    }
    const nonRenderedComponents = [];
    const expectedDocComponents = [
        'Main',
        'Head',
        'NextScript',
        'Html'
    ];
    for (const comp of expectedDocComponents){
        if (!docComponentsRendered[comp]) {
            nonRenderedComponents.push(comp);
        }
    }
    if (nonRenderedComponents.length) {
        const missingComponentList = nonRenderedComponents.map((e)=>`<${e} />`
        ).join(', ');
        const plural = nonRenderedComponents.length !== 1 ? 's' : '';
        throw new Error(`Your custom Document (pages/_document) did not render all the required subcomponent${plural}.\n` + `Missing component${plural}: ${missingComponentList}\n` + 'Read how to fix here: https://nextjs.org/docs/messages/missing-document-component');
    }
    const [renderTargetPrefix, renderTargetSuffix] = documentHTML.split(/<next-js-internal-body-render-target><\/next-js-internal-body-render-target>/);
    const prefix = [];
    if (!documentHTML.startsWith(DOCTYPE)) {
        prefix.push(DOCTYPE);
    }
    prefix.push(renderTargetPrefix);
    if (inAmpMode) {
        prefix.push('<!-- __NEXT_DATA__ -->');
    }
    let pipers = [
        piperFromArray(prefix),
        await documentResult.bodyResult(),
        piperFromArray([
            renderTargetSuffix
        ]), 
    ];
    const postProcessors = (generateStaticHTML ? [
        inAmpMode ? async (html)=>{
            html = await optimizeAmp(html, renderOpts.ampOptimizerConfig);
            if (!renderOpts.ampSkipValidation && renderOpts.ampValidator) {
                await renderOpts.ampValidator(html, pathname);
            }
            return html;
        } : null,
        !process.browser && (process.env.__NEXT_OPTIMIZE_FONTS || process.env.__NEXT_OPTIMIZE_IMAGES) ? async (html)=>{
            return await postProcess(html, {
                getFontDefinition
            }, {
                optimizeFonts: renderOpts.optimizeFonts,
                optimizeImages: renderOpts.optimizeImages
            });
        } : null,
        !process.browser && renderOpts.optimizeCss ? async (html)=>{
            // eslint-disable-next-line import/no-extraneous-dependencies
            const Critters = require('critters');
            const cssOptimizer = new Critters({
                ssrMode: true,
                reduceInlineStyles: false,
                path: renderOpts.distDir,
                publicPath: `${renderOpts.assetPrefix}/_next/`,
                preload: 'media',
                fonts: false,
                ...renderOpts.optimizeCss
            });
            return await cssOptimizer.process(html);
        } : null,
        inAmpMode || hybridAmp ? async (html)=>{
            return html.replace(/&amp;amp=1/g, '&amp=1');
        } : null, 
    ] : []).filter(Boolean);
    if (generateStaticHTML || postProcessors.length > 0) {
        let html = await piperToString(chainPipers(pipers));
        for (const postProcessor of postProcessors){
            if (postProcessor) {
                html = await postProcessor(html);
            }
        }
        return new _renderResult.default(html);
    }
    return new _renderResult.default(chainPipers(pipers));
}
function errorToJSON(err) {
    return {
        name: err.name,
        message: err.message,
        stack: err.stack,
        middleware: err.middleware
    };
}
function serializeError(dev, err) {
    if (dev) {
        return errorToJSON(err);
    }
    return {
        name: 'Internal Server Error.',
        message: '500 - Internal Server Error.',
        statusCode: 500
    };
}
function renderToNodeStream(element, generateStaticHTML) {
    return new Promise((resolve, reject)=>{
        let underlyingStream = null;
        const stream = new Writable({
            // Use the buffer from the underlying stream
            highWaterMark: 0,
            writev (chunks, callback) {
                let str = '';
                for (let { chunk  } of chunks){
                    str += chunk.toString();
                }
                if (!underlyingStream) {
                    throw new Error('invariant: write called without an underlying stream. This is a bug in Next.js');
                }
                if (!underlyingStream.writable.write(str)) {
                    underlyingStream.queuedCallbacks.push(()=>callback()
                    );
                } else {
                    callback();
                }
            }
        });
        stream.once('finish', ()=>{
            if (!underlyingStream) {
                throw new Error('invariant: finish called without an underlying stream. This is a bug in Next.js');
            }
            underlyingStream.resolve();
        });
        stream.once('error', (err)=>{
            if (!underlyingStream) {
                throw new Error('invariant: error called without an underlying stream. This is a bug in Next.js');
            }
            underlyingStream.resolve(err);
        });
        // React uses `flush` to prevent stream middleware like gzip from buffering to the
        // point of harming streaming performance, so we make sure to expose it and forward it.
        // See: https://github.com/reactwg/react-18/discussions/91
        Object.defineProperty(stream, 'flush', {
            value: ()=>{
                if (!underlyingStream) {
                    throw new Error('invariant: flush called without an underlying stream. This is a bug in Next.js');
                }
                if (typeof underlyingStream.writable.flush === 'function') {
                    underlyingStream.writable.flush();
                }
            },
            enumerable: true
        });
        let resolved = false;
        const doResolve = (startWriting)=>{
            if (!resolved) {
                resolved = true;
                resolve((res, next)=>{
                    const drainHandler = ()=>{
                        const prevCallbacks = underlyingStream.queuedCallbacks;
                        underlyingStream.queuedCallbacks = [];
                        prevCallbacks.forEach((callback)=>callback()
                        );
                    };
                    res.on('drain', drainHandler);
                    underlyingStream = {
                        resolve: (err)=>{
                            underlyingStream = null;
                            res.removeListener('drain', drainHandler);
                            next(err);
                        },
                        writable: res,
                        queuedCallbacks: []
                    };
                    startWriting();
                });
            }
        };
        const { abort , pipe  } = _server.default.renderToPipeableStream(element, {
            onError (error) {
                if (!resolved) {
                    resolved = true;
                    reject(error);
                }
                abort();
            },
            onCompleteShell () {
                if (!generateStaticHTML) {
                    doResolve(()=>pipe(stream)
                    );
                }
            },
            onCompleteAll () {
                doResolve(()=>pipe(stream)
                );
            }
        });
    });
}
async function bufferedReadFromReadableStream(reader, writeFn) {
    const decoder = new TextDecoder();
    let bufferedString = '';
    let pendingFlush = null;
    const flushBuffer = ()=>{
        if (!pendingFlush) {
            pendingFlush = new Promise((resolve)=>setTimeout(()=>{
                    writeFn(bufferedString);
                    bufferedString = '';
                    pendingFlush = null;
                    resolve();
                }, 0)
            );
        }
    };
    while(true){
        const { done , value  } = await reader.read();
        if (done) {
            break;
        }
        bufferedString += typeof value === 'string' ? value : decoder.decode(value);
        flushBuffer();
    }
    // Make sure the promise resolves after any pending flushes
    await pendingFlush;
}
function renderToWebStream(element) {
    return new Promise((resolve, reject)=>{
        let resolved = false;
        const stream = _server.default.renderToReadableStream(element, {
            onError (err) {
                if (!resolved) {
                    resolved = true;
                    reject(err);
                }
            },
            onCompleteShell () {
                if (!resolved) {
                    resolved = true;
                    resolve((res, next)=>{
                        bufferedReadFromReadableStream(reader, (val)=>res.write(val)
                        ).then(()=>next()
                        , (err)=>next(err)
                        );
                    });
                }
            }
        });
        const reader = stream.getReader();
    });
}
function chainPipers(pipers) {
    return pipers.reduceRight((lhs, rhs)=>(res, next)=>{
            rhs(res, (err)=>err ? next(err) : lhs(res, next)
            );
        }
    , (res, next)=>{
        res.end();
        next();
    });
}
function piperFromArray(chunks) {
    return (res, next)=>{
        if (typeof res.cork === 'function') {
            res.cork();
        }
        chunks.forEach((chunk)=>res.write(chunk)
        );
        if (typeof res.uncork === 'function') {
            res.uncork();
        }
        next();
    };
}
function piperToString(input) {
    return new Promise((resolve, reject)=>{
        const bufferedChunks = [];
        const stream = new Writable({
            writev (chunks, callback) {
                chunks.forEach((chunk)=>bufferedChunks.push(chunk.chunk)
                );
                callback();
            }
        });
        input(stream, (err)=>{
            if (err) {
                reject(err);
            } else {
                resolve(Buffer.concat(bufferedChunks).toString());
            }
        });
    });
}
function useMaybeDeferContent(_name, contentFn) {
    return [
        false,
        contentFn()
    ];
}

//# sourceMappingURL=render.js.map