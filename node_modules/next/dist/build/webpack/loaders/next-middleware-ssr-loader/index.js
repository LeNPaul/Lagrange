"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = middlewareSSRLoader;
var _stringifyRequest = require("../../stringify-request");
async function middlewareSSRLoader() {
    const { absolutePagePath , absoluteAppPath , absoluteDocumentPath , absolute500Path , absoluteErrorPath , isServerComponent , ...restRenderOpts } = this.getOptions();
    const stringifiedAbsolutePagePath = (0, _stringifyRequest).stringifyRequest(this, absolutePagePath);
    const stringifiedAbsoluteAppPath = (0, _stringifyRequest).stringifyRequest(this, absoluteAppPath);
    const stringifiedAbsolute500PagePath = (0, _stringifyRequest).stringifyRequest(this, absolute500Path || absoluteErrorPath);
    const stringifiedAbsoluteDocumentPath = (0, _stringifyRequest).stringifyRequest(this, absoluteDocumentPath);
    const transformed = `
    import { adapter } from 'next/dist/server/web/adapter'
    import { RouterContext } from 'next/dist/shared/lib/router-context'

    import App from ${stringifiedAbsoluteAppPath}
    import Document from ${stringifiedAbsoluteDocumentPath}

    import { getRender } from 'next/dist/build/webpack/loaders/next-middleware-ssr-loader/render'

    const pageMod = require(${stringifiedAbsolutePagePath})
    const errorMod = require(${stringifiedAbsolute500PagePath})

    const buildManifest = self.__BUILD_MANIFEST
    const reactLoadableManifest = self.__REACT_LOADABLE_MANIFEST
    const rscManifest = self.__RSC_MANIFEST

    if (typeof pageMod.default !== 'function') {
      throw new Error('Your page must export a \`default\` component')
    }

    const render = getRender({
      App,
      Document,
      pageMod,
      errorMod,
      buildManifest,
      reactLoadableManifest,
      rscManifest,
      isServerComponent: ${JSON.stringify(isServerComponent)},
      restRenderOpts: ${JSON.stringify(restRenderOpts)}
    })

    export default function rscMiddleware(opts) {
      return adapter({
        ...opts,
        handler: render
      })
    }`;
    return transformed;
}

//# sourceMappingURL=index.js.map