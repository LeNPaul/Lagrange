/// <reference types="react" />
declare const VALID_LOADING_VALUES: readonly ["lazy", "eager", undefined];
declare type LoadingValue = typeof VALID_LOADING_VALUES[number];
export declare type ImageLoader = (resolverProps: ImageLoaderProps) => string;
export declare type ImageLoaderProps = {
    src: string;
    width: number;
    quality?: number;
};
declare const VALID_LAYOUT_VALUES: readonly ["fill", "fixed", "intrinsic", "responsive", undefined];
declare type LayoutValue = typeof VALID_LAYOUT_VALUES[number];
declare type PlaceholderValue = 'blur' | 'empty';
declare type OnLoadingComplete = (result: {
    naturalWidth: number;
    naturalHeight: number;
}) => void;
declare type ImgElementStyle = NonNullable<JSX.IntrinsicElements['img']['style']>;
interface StaticRequire {
    default: StaticImageData;
}
declare type StaticImport = StaticRequire | StaticImageData;
export declare type ImageProps = Omit<JSX.IntrinsicElements['img'], 'src' | 'srcSet' | 'ref' | 'width' | 'height' | 'loading' | 'style'> & {
    src: string | StaticImport;
    width?: number | string;
    height?: number | string;
    layout?: LayoutValue;
    loader?: ImageLoader;
    quality?: number | string;
    priority?: boolean;
    loading?: LoadingValue;
    lazyBoundary?: string;
    placeholder?: PlaceholderValue;
    blurDataURL?: string;
    unoptimized?: boolean;
    objectFit?: ImgElementStyle['objectFit'];
    objectPosition?: ImgElementStyle['objectPosition'];
    onLoadingComplete?: OnLoadingComplete;
};
export default function Image({ src, sizes, unoptimized, priority, loading, lazyBoundary, className, quality, width, height, objectFit, objectPosition, onLoadingComplete, loader, placeholder, blurDataURL, ...all }: ImageProps): JSX.Element;
export {};
