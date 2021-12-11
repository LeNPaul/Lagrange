import { IncrementalCache } from './incremental-cache';
import RenderResult from './render-result';
interface CachedRedirectValue {
    kind: 'REDIRECT';
    props: Object;
}
interface CachedPageValue {
    kind: 'PAGE';
    html: RenderResult;
    pageData: Object;
}
export declare type ResponseCacheValue = CachedRedirectValue | CachedPageValue;
export declare type ResponseCacheEntry = {
    revalidate?: number | false;
    value: ResponseCacheValue | null;
};
declare type ResponseGenerator = (hasResolved: boolean) => Promise<ResponseCacheEntry | null>;
export default class ResponseCache {
    incrementalCache: IncrementalCache;
    pendingResponses: Map<string, Promise<ResponseCacheEntry | null>>;
    constructor(incrementalCache: IncrementalCache);
    get(key: string | null, responseGenerator: ResponseGenerator): Promise<ResponseCacheEntry | null>;
}
export {};
