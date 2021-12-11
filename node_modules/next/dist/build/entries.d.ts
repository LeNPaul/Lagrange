import { __ApiPreviewProps } from '../server/api-utils';
import { LoadedEnvFiles } from '@next/env';
import { NextConfigComplete } from '../server/config-shared';
import type { webpack5 } from 'next/dist/compiled/webpack/webpack';
declare type ObjectValue<T> = T extends {
    [key: string]: infer V;
} ? V : never;
export declare type PagesMapping = {
    [page: string]: string;
};
export declare function createPagesMapping(pagePaths: string[], extensions: string[], { isDev, hasServerComponents, hasConcurrentFeatures, }: {
    isDev: boolean;
    hasServerComponents: boolean;
    hasConcurrentFeatures: boolean;
}): PagesMapping;
declare type Entrypoints = {
    client: webpack5.EntryObject;
    server: webpack5.EntryObject;
    serverWeb: webpack5.EntryObject;
};
export declare function createEntrypoints(pages: PagesMapping, target: 'server' | 'serverless' | 'experimental-serverless-trace', buildId: string, previewMode: __ApiPreviewProps, config: NextConfigComplete, loadedEnvFiles: LoadedEnvFiles): Entrypoints;
export declare function finalizeEntrypoint({ name, value, isServer, isMiddleware, isServerWeb, }: {
    isServer: boolean;
    name: string;
    value: ObjectValue<webpack5.EntryObject>;
    isMiddleware?: boolean;
    isServerWeb?: boolean;
}): ObjectValue<webpack5.EntryObject>;
export {};
