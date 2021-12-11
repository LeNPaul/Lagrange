import '@next/polyfill-module';
import { MittEmitter } from '../shared/lib/mitt';
import type Router from '../shared/lib/router/router';
import { AppComponent, PrivateRouteInfo } from '../shared/lib/router/router';
import { NEXT_DATA } from '../shared/lib/utils';
declare global {
    interface Window {
        __NEXT_HYDRATED?: boolean;
        __NEXT_HYDRATED_CB?: () => void;
        __NEXT_PRELOADREADY?: (ids?: (string | number)[]) => void;
        __NEXT_DATA__: NEXT_DATA;
        __NEXT_P: any[];
    }
}
declare type RenderRouteInfo = PrivateRouteInfo & {
    App: AppComponent;
    scroll?: {
        x: number;
        y: number;
    } | null;
};
declare type RenderErrorProps = Omit<RenderRouteInfo, 'Component' | 'styleSheets'>;
export declare const version: string | undefined;
export declare let router: Router;
export declare const emitter: MittEmitter<string>;
export declare function initNext(opts?: {
    webpackHMR?: any;
}): Promise<MittEmitter<string> | {
    emitter: MittEmitter<string>;
    renderCtx: Omit<import("../shared/lib/router/router").CompletePrivateRouteInfo, "styleSheets"> & {
        initial: true;
    } & {
        App: AppComponent;
        scroll?: {
            x: number;
            y: number;
        } | null | undefined;
    };
}>;
export declare function render(renderingProps: RenderRouteInfo): Promise<void>;
export declare function renderError(renderErrorProps: RenderErrorProps): Promise<any>;
export {};
