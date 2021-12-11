import { NextRequest } from '../../../../server/web/spec-extension/request';
export declare function getRender({ App, Document, pageMod, errorMod, rscManifest, buildManifest, reactLoadableManifest, isServerComponent, restRenderOpts, }: {
    App: any;
    Document: any;
    pageMod: any;
    errorMod: any;
    rscManifest: object;
    buildManifest: any;
    reactLoadableManifest: any;
    isServerComponent: boolean;
    restRenderOpts: any;
}): (request: NextRequest) => Promise<Response>;
