import { webpack } from 'next/dist/compiled/webpack/webpack';
import { NextConfigComplete } from '../../../server/config-shared';
export declare function build(config: webpack.Configuration, { supportedBrowsers, rootDirectory, customAppFile, isDevelopment, isServer, webServerRuntime, targetWeb, assetPrefix, sassOptions, productionBrowserSourceMaps, future, experimental, disableStaticImages, }: {
    supportedBrowsers: string[] | undefined;
    rootDirectory: string;
    customAppFile: RegExp;
    isDevelopment: boolean;
    isServer: boolean;
    webServerRuntime: boolean;
    targetWeb: boolean;
    assetPrefix: string;
    sassOptions: any;
    productionBrowserSourceMaps: boolean;
    future: NextConfigComplete['future'];
    experimental: NextConfigComplete['experimental'];
    disableStaticImages: NextConfigComplete['disableStaticImages'];
}): Promise<webpack.Configuration>;
