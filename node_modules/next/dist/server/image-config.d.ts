export declare const VALID_LOADERS: readonly ["default", "imgix", "cloudinary", "akamai", "custom"];
export declare type LoaderValue = typeof VALID_LOADERS[number];
declare type ImageFormat = 'image/avif' | 'image/webp';
export declare type ImageConfigComplete = {
    deviceSizes: number[];
    imageSizes: number[];
    loader: LoaderValue;
    path: string;
    domains?: string[];
    disableStaticImages?: boolean;
    minimumCacheTTL?: number;
    formats?: ImageFormat[];
};
export declare type ImageConfig = Partial<ImageConfigComplete>;
export declare const imageConfigDefault: ImageConfigComplete;
export {};
