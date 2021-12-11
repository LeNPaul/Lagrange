"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.getRender = getRender;
var _render = require("../../../../server/web/render");
var _utils = require("../../../../server/web/utils");
const createHeaders = (args)=>({
        ...args,
        'x-middleware-ssr': '1'
    })
;
function getRender({ App , Document , pageMod , errorMod , rscManifest , buildManifest , reactLoadableManifest , isServerComponent , restRenderOpts  }) {
    return async function render(request) {
        const { nextUrl: url , cookies , headers  } = request;
        const { pathname , searchParams  } = url;
        const query = Object.fromEntries(searchParams);
        // Preflight request
        if (request.method === 'HEAD') {
            return new Response(null, {
                headers: createHeaders()
            });
        }
        const renderServerComponentData = isServerComponent ? query.__flight__ !== undefined : false;
        delete query.__flight__;
        const req = {
            url: pathname,
            cookies,
            headers: (0, _utils).toNodeHeaders(headers)
        };
        const renderOpts = {
            ...restRenderOpts,
            // Locales are not supported yet.
            // locales: i18n?.locales,
            // locale: detectedLocale,
            // defaultLocale,
            // domainLocales: i18n?.domains,
            dev: process.env.NODE_ENV !== 'production',
            App,
            Document,
            buildManifest,
            Component: pageMod.default,
            pageConfig: pageMod.config || {
            },
            getStaticProps: pageMod.getStaticProps,
            getServerSideProps: pageMod.getServerSideProps,
            getStaticPaths: pageMod.getStaticPaths,
            reactLoadableManifest,
            env: process.env,
            supportsDynamicHTML: true,
            concurrentFeatures: true,
            renderServerComponentData,
            serverComponentManifest: isServerComponent ? rscManifest : null,
            ComponentMod: null
        };
        const transformStream = new TransformStream();
        const writer = transformStream.writable.getWriter();
        const encoder = new TextEncoder();
        let result;
        try {
            result = await (0, _render).renderToHTML(req, {
            }, pathname, query, renderOpts);
        } catch (err) {
            const errorRes = {
                statusCode: 500,
                err
            };
            try {
                result = await (0, _render).renderToHTML(req, errorRes, '/_error', query, {
                    ...renderOpts,
                    Component: errorMod.default,
                    getStaticProps: errorMod.getStaticProps,
                    getServerSideProps: errorMod.getServerSideProps,
                    getStaticPaths: errorMod.getStaticPaths
                });
            } catch (err2) {
                return new Response((err2 || 'An error occurred while rendering ' + pathname + '.').toString(), {
                    status: 500,
                    headers: createHeaders()
                });
            }
        }
        if (!result) {
            return new Response('An error occurred while rendering ' + pathname + '.', {
                status: 500,
                headers: createHeaders()
            });
        }
        result.pipe({
            write: (str)=>writer.write(encoder.encode(str))
            ,
            end: ()=>writer.close()
        });
        return new Response(transformStream.readable, {
            headers: createHeaders(),
            status: 200
        });
    };
}

//# sourceMappingURL=render.js.map