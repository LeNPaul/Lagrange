/// <reference types="lru-cache" />
import LRUCache from 'next/dist/compiled/lru-cache';
import { PrerenderManifest } from '../build';
interface CachedRedirectValue {
    kind: 'REDIRECT';
    props: Object;
}
interface CachedPageValue {
    kind: 'PAGE';
    html: string;
    pageData: Object;
}
export declare type IncrementalCacheValue = CachedRedirectValue | CachedPageValue;
declare type IncrementalCacheEntry = {
    curRevalidate?: number | false;
    revalidateAfter: number | false;
    isStale?: boolean;
    value: IncrementalCacheValue | null;
};
export declare class IncrementalCache {
    incrementalOptions: {
        flushToDisk?: boolean;
        pagesDir?: string;
        distDir?: string;
        dev?: boolean;
    };
    prerenderManifest: PrerenderManifest;
    cache?: LRUCache<string, IncrementalCacheEntry>;
    locales?: string[];
    constructor({ max, dev, distDir, pagesDir, flushToDisk, locales, }: {
        dev: boolean;
        max?: number;
        distDir: string;
        pagesDir: string;
        flushToDisk?: boolean;
        locales?: string[];
    });
    private getSeedPath;
    private calculateRevalidate;
    getFallback(page: string): Promise<string>;
    get(pathname: string): Promise<IncrementalCacheEntry | null>;
    set(pathname: string, data: IncrementalCacheValue | null, revalidateSeconds?: number | false): Promise<void>;
}
export {};
